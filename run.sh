

# train
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 200 --batch_size 256 --lr 0.1  --dataset cifar100 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_cifart100/
CUDA_VISIBLE_DEVICES=2 python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 200 --batch_size 256 --lr 0.1  --dataset cifar10 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_cifart10/
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name resnet_50_xrays --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 64 --lr 0.1  --dataset xray --data_dir $HOME/data/ --log_freq 50 --checkpoint_dir ./checkpoint_xrays/

# infernce
CUDA_VISIBLE_DEVICES=1 python -W ignore  inference.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 1 --batch_size 512 --lr 0.1  --dataset cifar10 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_cifart10/ --resume_best

CUDA_VISIBLE_DEVICES=1 python -W ignore  inference.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 1 --batch_size 512 --lr 0.1  --dataset cifar100 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_cifart100/ --resume_best


scp -r train_xray xingnaili@ncrs.d2.comp.nus.edu.sg:/home/xingnaili/data/xrayData

scp -r val_xray xingnaili@ncrs.d2.comp.nus.edu.sg:/home/xingnaili/data/xrayData



This research is supported by Singapore Ministry of Education Academic Research Fund Tier 3 under MOEs official grant number MOE2017-T3-1-007.
