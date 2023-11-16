from default_config import basic_cfg
import os
import albumentations as A
cfg = basic_cfg

cfg.debug = 0
cfg.out_dir = f"outputs/{os.path.basename(__file__).split('.')[0]}"

cfg.in_w = 2048 
cfg.in_h = 128
cfg.batch_size = 32
cfg.train_epoch = 200
cfg.lr = 2e-4
cfg.backbone = 'tf_efficientnetv2_b2.in1k'
cfg.hidden_dim = 256
cfg.fold=5
cfg.load_weight = 'outputs/b2_256_ptr_f5/b2_256_ptr_f5_last.pth'

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