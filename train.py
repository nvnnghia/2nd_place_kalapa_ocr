import os
import gc
import cv2
import sys
import math
import timm
import torch
import random
import argparse
import importlib
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch.nn as nn
from shutil import copyfile
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import albumentations as A
from collections import OrderedDict
from warnings import filterwarnings
filterwarnings("ignore")

sys.path.append("configs")

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("-M", "--mode", default='train', help="mode type")
parser_args = parser.parse_args()

print("[ √ ] Using config file", parser_args.config)
print("[ √ ] Using mode: ", parser_args.mode)

cfg = importlib.import_module(parser_args.config).cfg

cfg.mode = parser_args.mode
cfg.config = parser_args.config

mode = parser_args.mode

def logfile(message):
    print(message)
    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')


from imgaug import augmenters as iaa
import imgaug as ia

class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)

    self.aug = iaa.Sequential(iaa.SomeOf((1, 5), 
        [

        sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)),
                            iaa.MotionBlur(k=3)])),
        
        # color
        sometimes(iaa.Invert(0.25, per_channel=0.5)),
        sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
        sometimes(iaa.Dropout2d(p=0.5)),

        sometimes(iaa.JpegCompression(compression=(5, 80))),
        
        # distort
        sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
        sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), 
#                            rotate=(-5, 5), shear=(-5, 5), 
                            order=[0, 1], cval=(0, 255), 
                            mode=ia.ALL)),

    ],
        random_order=True),
    random_order=True)
      
  def __call__(self, img):
    img = self.aug.augment_image(img)
    return img

img_tfms = ImgAugTransform()

class OCRDataset(Dataset):
    def __init__(self, df, vocab, transform=None):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        if cfg.is_pretrain:
            img_path = row['img_path']
        else:
            img_path = 'training_data/images/' + row['img_path']
        
        label = row['text']

        img_bw = cv2.imread(img_path)
        img_bw = self.transform(image=img_bw)["image"]

        if len(self.df) > 1000:
            img_bw = img_tfms(img_bw)

        img_bw = img_bw.transpose(2,0,1)
        img_bw = img_bw/255 

        word = self.vocab.encode(label)

        sample = {'img': img_bw, 'word': word, 'img_path': img_path}

        return sample

    def __len__(self):
        return len(self.df)


class Collator(object):
    def __init__(self, masked_language_model=True):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        filenames = []
        img = []
        target_weights = []
        tgt_input = []
        max_label_len = max(len(sample['word']) for sample in batch)
        max_img_len = max(sample['img'].shape[-1] for sample in batch )
        for sample in batch:
            im_c, im_h, im_w = sample['img'].shape
            temp = np.zeros((im_c,im_h,max_img_len))
            temp[:,:,:im_w] = sample['img']
            img.append(temp)
            filenames.append(sample['img_path'])
            label = sample['word']
            label_len = len(label)
            
            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len,dtype=np.float32))))
            
        img = np.array(img, dtype=np.float32)

        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1] = 0
        
        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights)==0

        rs = {
            'img': torch.FloatTensor(img),
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask),
            'filenames': filenames
        }   
        
        return rs

class Timm(nn.Module):
    def __init__(self, name, hidden = 256, drop_path_rate=0.1, drop_rate=0.1, dropout=0.5):
        super(Timm, self).__init__()

        base_model = timm.create_model(name, pretrained=True, 
            drop_rate = drop_rate,
            drop_path_rate =drop_path_rate
            )

        layers = list(base_model.children())[:-2]

        self.encoder = nn.Sequential(*layers)

        in_features = base_model.num_features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(in_features, hidden, 1)

    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        conv = self.encoder(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)

        conv = conv.permute(-1, 0, 1)


        return conv


class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
                
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        src: src_len x batch_size x img_channel
        outputs: src_len x batch_size x hid_dim 
        hidden: batch_size x hid_dim
        """

        embedded = self.dropout(src)
        
        outputs, hidden = self.rnn(embedded)
                                 
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim,
        outputs: batch_size x src_len
        """
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        """
        inputs: batch_size
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """
             
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs)
                
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, encoder_hidden, decoder_hidden, img_channel, decoder_embedded, dropout=0.1):
        super().__init__()
        
        attn = Attention(encoder_hidden, decoder_hidden)
        
        self.encoder = Encoder(img_channel, encoder_hidden, decoder_hidden, dropout)
        self.decoder = Decoder(vocab_size, decoder_embedded, encoder_hidden, decoder_hidden, dropout, attn)
        
    def forward_encoder(self, src):       
        """
        src: timestep x batch_size x channel
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        encoder_outputs, hidden = self.encoder(src)

        return (hidden, encoder_outputs)

    def forward_decoder(self, tgt, memory):
        """
        tgt: timestep x batch_size 
        hidden: batch_size x hid_dim
        encouder: src_len x batch_size x hid_dim
        output: batch_size x 1 x vocab_size
        """
        
        tgt = tgt[-1]
        hidden, encoder_outputs = memory
        output, hidden, _ = self.decoder(tgt, hidden, encoder_outputs)
        output = output.unsqueeze(1)
        
        return output, (hidden, encoder_outputs)

    def forward(self, src, trg):
        """
        src: time_step x batch_size
        trg: time_step x batch_size
        outputs: batch_size x time_step x vocab_size
        """

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        device = src.device

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src)
                
        ##TODO reverse the order>> decode backward??
        for t in range(trg_len):
            input = trg[t] 
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            outputs[t] = output
            
        outputs = outputs.transpose(0, 1).contiguous()

        return outputs

class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c:i+4 for i, c in enumerate(chars)}

        self.i2c = {i+4:c for i, c in enumerate(chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars if c in self.c2i] + [self.eos]
    
    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars

class VietOCR(nn.Module):
    def __init__(self, vocab):
        
        super(VietOCR, self).__init__()

        self.vocab = vocab

        hidden_dim = cfg.hidden_dim
        self.cnn = Timm(cfg.backbone, hidden = hidden_dim, drop_path_rate=0.5, drop_rate=0.5, dropout=0.5)

        self.transformer = Seq2Seq(len(self.vocab), encoder_hidden=hidden_dim, decoder_hidden=hidden_dim, img_channel=hidden_dim, decoder_embedded=hidden_dim, dropout=0.5) 

    def forward(self, img, tgt_input, tgt_key_padding_mask=None):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(src, tgt_input)

        return outputs

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

from Levenshtein import distance
import unicodedata
def dist_calc(str1, str2):
    return distance(str1, str2)


import json 
if __name__ == '__main__':
    out_dir = cfg.out_dir
    out_name = out_dir.split('/')[-1]
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = f'{cfg.out_dir}/log.txt'

    if cfg.mode == 'train':
        copyfile(os.path.basename(__file__), os.path.join(cfg.out_dir, os.path.basename(__file__)))
        copyfile(f'configs/{cfg.config}.py', os.path.join(cfg.out_dir, f'{cfg.config}.py'))

    if cfg.is_pretrain:
        df = pd.read_csv('train_ext3.csv')
        train_df = df[df.fold!=cfg.fold]

        val_df = df[df.fold==cfg.fold]
        val_df = val_df[val_df.aug==0]
    else:
        df = pd.read_csv('train_folds.csv')
        train_df = df[df.fold!=cfg.fold]
        val_df = df[df.fold==cfg.fold] 
        if cfg.fold > 4:
            val_df = df[df.fold==0] 

    if val_df.shape[0]<10:
        val_df = df.head(100)

    print(train_df.shape, val_df.shape)

    vocab =  'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~ '
    vocab = Vocab(vocab)
    model = VietOCR(vocab)


    device = 'cuda'
    model.to(device)

    if cfg.mode == 'train': #train
        if len(cfg.load_weight) > 0:
            print(f' load {cfg.load_weight}!')
            checkpoint = torch.load(cfg.load_weight, map_location="cpu")
            model.load_state_dict(checkpoint)


        train_dataset = OCRDataset(train_df, vocab, transform=cfg.train_transform)
        collate_fn = Collator(masked_language_model=True)
        train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size, 
                collate_fn = collate_fn,
                shuffle=True,
                drop_last=True,
                num_workers=8,
                )

        val_dataset = OCRDataset(val_df, vocab, transform=cfg.val_transform)
        val_collate_fn = Collator(masked_language_model=False)
        val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size, 
                collate_fn = val_collate_fn,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-09)
        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=cfg.train_epoch*len(train_loader),
            )
        iter_update = True
        criterion = LabelSmoothingLoss(len(vocab), padding_idx=vocab.pad, smoothing=cfg.smooth)

        best_loss = 10
        global_step = 0
        for epoch in range(1,cfg.train_epoch+1):
            if not iter_update:
                scheduler.step(epoch)

            scaler = torch.cuda.amp.GradScaler(enabled=False)

            model.train()
            losses = []
            bar = tqdm(train_loader)
            for batch_idx, batch in enumerate(bar):
                img = batch['img'].to(device, non_blocking=True)
                tgt_input = batch['tgt_input'].to(device, non_blocking=True)
                tgt_output = batch['tgt_output'].to(device, non_blocking=True)
                tgt_padding_mask = batch['tgt_padding_mask'].to(device, non_blocking=True)

                outputs = model(img, tgt_input, tgt_padding_mask)
                outputs = outputs.view(-1, outputs.size(2))#flatten(0, 1)
                tgt_output = tgt_output.view(-1)#flatten()
                
                loss = criterion(outputs, tgt_output)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if iter_update:
                    scheduler.step()

                losses.append(loss.item())
                smooth_loss = np.mean(losses[:])

                bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}, LR {scheduler.get_lr()[0]:.6f}')

                global_step +=1

            train_loss = np.mean(losses)

            #eval 
            val_losses = []
            model.eval()
            val_bar = tqdm(val_loader)
            for batch_idx, batch in enumerate(val_bar):
                img = batch['img'].to(device, non_blocking=True)
                tgt_input = batch['tgt_input'].to(device, non_blocking=True)
                tgt_output = batch['tgt_output'].to(device, non_blocking=True)
                tgt_padding_mask = batch['tgt_padding_mask'].to(device, non_blocking=True)
                with torch.no_grad():
                    outputs = model(img, tgt_input, tgt_padding_mask)
                outputs = outputs.view(-1, outputs.size(2))#flatten(0, 1)
                tgt_output = tgt_output.view(-1)#flatten()
                
                loss = criterion(outputs, tgt_output)
                val_losses.append(loss.item())
                val_bar.set_description(f'loss: {loss.item():.5f}, smth: {np.mean(val_losses):.5f}')

            val_loss = np.mean(val_losses)
            logfile(f'[EPOCH] {epoch}, train_loss: {train_loss:.6f},  Val loss: {val_loss:.6f}')
            if val_loss < best_loss:
                logfile(f'[EPOCH] {epoch} ===============> best loss ({best_loss:.6f} --> {val_loss:.6f}). Saving model .......!!!!\n')
                # torch.save(model.state_dict(), f'{out_dir}/best_loss.pth')
                best_loss = val_loss

            if epoch>99 and epoch%5==0:
                torch.save(model.state_dict(), f'{out_dir}/{out_name}_ep{epoch}.pth')

            # torch.save(model.state_dict(), f'{out_dir}/{out_name}_last.pth')

        # torch.save(model.half().state_dict(), f'{out_dir}/{out_name}_last.pth')

        #postprocess weight
        if not cfg.is_pretrain:
            count = 0
            swa_chkp = OrderedDict({"state_dict": None})
            for epoch in range(100,cfg.train_epoch+1,5):
                count +=1
                path = f'{out_dir}/{out_name}_ep{epoch}.pth'
                temp_chkp = torch.load(path, map_location="cpu")
                if swa_chkp['state_dict'] is None:
                    swa_chkp['state_dict'] = temp_chkp
                else:
                    for k in swa_chkp['state_dict'].keys():
                        if isinstance(swa_chkp['state_dict'][k], torch.FloatTensor):
                            swa_chkp['state_dict'][k] += temp_chkp[k]

                if os.path.isfile(path):
                    os.remove(path)

            for k in swa_chkp['state_dict'].keys():
                if isinstance(swa_chkp['state_dict'][k], torch.FloatTensor):
                    swa_chkp['state_dict'][k] = (swa_chkp['state_dict'][k]/count).half()
            torch.save(swa_chkp['state_dict'], f'{out_dir}/{out_name}_swa.pth')
        else:
            torch.save(model.half().state_dict(), f'{out_dir}/{out_name}_last.pth')


