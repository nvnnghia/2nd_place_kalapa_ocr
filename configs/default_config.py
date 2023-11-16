import os
from types import SimpleNamespace
import albumentations as A

cfg = SimpleNamespace(**{})

cfg.debug = 0
cfg.train_csv_path = "data/train_folds.csv"
cfg.out_dir = f"outputs/{os.path.basename(__file__).split('.')[0]}"
cfg.load_weight = ''
cfg.is_pretrain = 0 

cfg.image_height = 32 
cfg.image_min_width = 32 
cfg.image_max_width = 512
cfg.lr = 1e-3

cfg.in_w = 2048 
cfg.in_h = 128
cfg.batch_size = 32
cfg.train_epoch = 10
cfg.fold = 0 
cfg.is_swa = 0 
cfg.backbone = 'tf_efficientnetv2_b1.in1k'
cfg.hidden_dim = 384
cfg.smooth = 0.1

cfg.train_transform = A.Compose([
        A.Resize(height=cfg.in_h, width=cfg.in_w, interpolation=1, p=1),
        A.OneOf([
            A.RandomBrightness(always_apply=False, p=1.0, limit=0.2),
            A.RandomContrast(always_apply=False, p=1.0, limit=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.5),
    ])
cfg.val_transform = A.Compose([
        A.Resize(height=cfg.in_h, width=cfg.in_w, interpolation=1, p=1),
    ])

basic_cfg = cfg