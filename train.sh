# dataset selection:
# set data_ind (0, 1, 2) to select dataset in data_dir_list
# testing checkpoint
# --resume_checkpoint logs/2023-12-15-21-52-50-mpi3d_real_complex_FDAE_seed0_rank0/model100000.pt --eval_only True
gpu=0
data_ind=0
data_dir_list=(datasets/shapes3d datasets/cars3d datasets/mpi3d_real_complex)
group_num_list=(6 2 6)
data_dir=${data_dir_list[${data_ind}]}
group_num=${group_num_list[${data_ind}]}

content_decorrelation_weight=2.5e-5
mask_entropy_weight=1.0e-4

max_step=100000
eval_interval=100000
save_interval=100000
code_dim=80
batch=32
encoder_type="resnet18"
for seed in 0
do
python fdae_train.py --available_gpus ${gpu} \
--log_suffix FDAE_seed${seed}_ \
--data_dir ${data_dir} \
--mask_entropy_weight ${mask_entropy_weight} --content_decorrelation_weight ${content_decorrelation_weight} \
--semantic_group_num ${group_num} --semantic_code_dim ${code_dim} --mask_code_dim ${code_dim} --semantic_code_adjust_dim ${code_dim} \
--additional_cond_map_layer_dim ${code_dim} \
--max_step ${max_step} --save_interval ${save_interval} --eval_interval ${eval_interval} --attention_resolutions 32,16,8 \
--global_batch_size ${batch} --lr 0.0001 \
--class_cond False --image_cond True --use_scale_shift_norm False --dropout 0.1 --image_size 64 \
--num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal \
--use_fp16 True --weight_decay 0.0 --weight_schedule karras --debug_mode True \
--seed ${seed} \
--encoder_type ${encoder_type}
done