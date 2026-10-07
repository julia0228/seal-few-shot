python train.py -batch 128 -dataset miniimagenet -gpu 0 -shot 5 -query 15 -unlabeled 15 -pseudo_topk 5 -extra_dir seal-5w5s -temperature_attn 5.0 -lamb 0.25 -milestones 40 50 -max_epoch 60 -no_wandb
