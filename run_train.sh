# -- fixed mask --
CUDA_VISIBLE_DEVICES=3 python train.py \
--batch_size 4 \
--learning_rate 2e-5 \
--image_size 512 512 \
--num_stage 10 \
--cs_ratio 4 4 \
--warmup_steps 5 \
--end_epoch 200 \
--data_path /data2/fanzhuoyao/MyProjects/BMNet_modified/dataset/train/ \
--save_dir ./model_ckpt/ \
--local_rank -1 \
--use_checkpoint

# -- learnable mask ---
# CUDA_VISIBLE_DEVICES=3 python train.py --batch_size 4 --learning_rate 2e-5 --image_size 512 512 --num_stage 10 --cs_ratio 4 4 --warmup_steps 5 --end_epoch 100 --data_path /data2/fanzhuoyao/MyProjects/BMNet_modified/dataset/train/ --save_dir ./model_ckpt/ --local_rank -1 --lm
