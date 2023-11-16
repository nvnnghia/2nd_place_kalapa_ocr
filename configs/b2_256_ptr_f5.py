from default_config import basic_cfg
import os
cfg = basic_cfg

cfg.debug = 0
cfg.out_dir = f"outputs/{os.path.basename(__file__).split('.')[0]}"

cfg.in_w = 2048 
cfg.in_h = 128
cfg.batch_size = 32
cfg.train_epoch = 30
# cfg.lr = 3e-4
cfg.backbone = 'tf_efficientnetv2_b2.in1k'
cfg.hidden_dim = 256
cfg.fold=5
cfg.is_pretrain = 1
