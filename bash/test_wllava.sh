CUDA_VISIBLE_DEVICES=0,1 python /PATH/DiT4SR-main/test/test_wllava.py \
--pretrained_model_name_or_path="/PATH/DiT4SR-main/preset/models/stable-diffusion-3.5-medium" \
--transformer_model_name_or_path="/PATH/checkpoint-1000/transformer" \
--image_path /PATH/Data/SR/RealLQ250/lq \
--output_dir /PATH/testpic \
# --save_prompts \
