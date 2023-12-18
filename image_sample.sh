# save masks for images in --data_dir
# set save_mask=True
python conditional_image_sample.py \
--model_path logs/2023-12-15-21-51-24-shapes3d_FDAE_seed0_rank0/model100000.pt \
--seed 0 \
--num_samples 5 \
--batch_size 5 \
--data_dir visualization/shapes3d_test \
--save_mask True \
--class_cond False \
--sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --weight_schedule karras \
--attention_resolutions 32,16,8 --use_scale_shift_norm False --dropout 0.0 --image_size 64 \
--num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True

# swapping content codes or mask codes for each pair of images in --data_dir
# set swap_flag=True
# for factor_dim, c1,2-m0 denotes swapping content code 1 and 2, c0-m1,2 denotes swapping mask codes 1 and 2
python conditional_image_sample.py \
--model_path logs/2023-12-15-21-51-24-shapes3d_FDAE_seed0_rank0/model100000.pt \
--swap_flag True --factor_dim c1,2,3,4,5,6-m0 \
--seed 0 \
--num_samples 5 \
--batch_size 5 \
--data_dir visualization/shapes3d_test \
--class_cond False \
--sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --weight_schedule karras \
--attention_resolutions 32,16,8 --use_scale_shift_norm False --dropout 0.0 --image_size 64 \
--num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True
