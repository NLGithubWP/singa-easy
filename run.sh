

python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 256 --lr 0.1  --dataset cifar100 --data_dir $HOME/data/ --log_freq 50
python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 1 --batch_size 256 --lr 0.1  --dataset cifar10 --data_dir $HOME/data/ --log_freq 50 --checkpoint_dir ./checkpoint/

python inference.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 1 --batch_size 256 --lr 0.1  --dataset cifar10 --data_dir $HOME/data/ --log_freq 50 --resume checkpoint --checkpoint_dir ./checkpoint/ --resume_best
