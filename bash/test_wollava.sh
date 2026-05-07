

CUDA_VISIBLE_DEVICES=0 python /PATH/DiT4SR-main/test/test_wollava.py \
--pretrained_model_name_or_path="/PATH/DiT4SR-main/preset/models/stable-diffusion-3.5-medium" \
--transformer_model_name_or_path="/PATH/checkpoint-1500/transformer_target" \
--image_path /PATH/Data/SR/RealLQ250/lq \
--output_dir /PATH/testpic/RealLQ250 \
--prompt_path /PATH/Data/SR/llavaCaptionRealLQ250/txt/ \
