

# train
#CUDA_VISIBLE_DEVICES=1 python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 400 --batch_size 256 --lr 0.05  --dataset cifar100 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_cifart100/
#CUDA_VISIBLE_DEVICES=2 python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 200 --batch_size 256 --lr 0.1  --dataset cifar10 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_cifart10/

# xray 1. origin 2. checkpoint
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name resnet_50_xrays --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 64 --lr 0.1  --dataset xray --data_dir $HOME/data/ --log_freq 50 --checkpoint_dir ./checkpoint_xrays/
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name resnet_50_xrays --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 64 --lr 0.1  --dataset xray --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_xrays/

#food 1. origin 2. checkpoint
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name resnet_50_food --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 300 --batch_size 64 --lr 0.1  --dataset food --data_dir $HOME/data/ --log_freq 50 --checkpoint_dir ./checkpoint_food/
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name resnet_50_food --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 300 --batch_size 64 --lr 0.1  --dataset food --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_food/

#image net
CUDA_VISIBLE_DEVICES=0 python train.py --exp_name resnet_50_imgnet --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 64 --lr 0.1  --dataset imagenet --data_dir /hdd1/xingnaili/ --log_freq 50 --checkpoint_dir ./checkpoint_imagenet/
CUDA_VISIBLE_DEVICES=1 python train.py --exp_name resnet_50_imgnet --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 300 --batch_size 64 --lr 0.1  --dataset imagenet --data_dir /hdd1/xingnaili/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_imagenet/


# infernce
CUDA_VISIBLE_DEVICES=0 python -W ignore  inference.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 1 --batch_size 512 --lr 0.1  --dataset cifar10 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_cifart10/ --resume_best
CUDA_VISIBLE_DEVICES=1 python -W ignore  inference.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 1 --batch_size 512 --lr 0.1  --dataset cifar100 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_cifart100/ --resume_best

#xray
CUDA_VISIBLE_DEVICES=0 python -W ignore inference.py --exp_name resnet_50_xrays --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 1 --batch_size 1 --lr 0.1  --dataset xray --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_xrays/ --resume_best --predict_image_nums 1

#food
CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py --exp_name resnet_50_food --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 32 --lr 0.1  --dataset food --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_food/ --resume_best
CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py --exp_name resnet_50_food --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 32 --lr 0.1  --dataset food --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_food/ --resume_best

#mock image-net
CUDA_VISIBLE_DEVICES=1 python -W ignore inference.py --exp_name resnet_50_xrays --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 1 --batch_size 32 --lr 0.1  --dataset imgTest --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint_xrays/ --resume_best

scp -r train_xray xingnaili@ncrs.d2.comp.nus.edu.sg:/home/xingnaili/data/xrayData
scp -r val_xray xingnaili@ncrs.d2.comp.nus.edu.sg:/home/xingnaili/data/xrayData

