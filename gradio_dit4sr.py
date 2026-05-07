import gradio as gr
from typing import List
import argparse
import sys
import os
import glob
sys.path.append(os.getcwd())
from llava.llm_agent import LLavaAgent
from PIL import Image
from CKPT_PTH import LLAVA_MODEL_PATH
import re

import numpy as np
from PIL import Image

import torch
from pytorch_lightning import seed_everything
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline

from utils.wavelet_color_fix import adain_color_fix

from torchvision import transforms
from model_dit4sr.transformer_sd3 import SD3Transformer2DModel

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", type=str, default='preset/models/stable-diffusion-3.5-medium')
parser.add_argument("--transformer_model_name_or_path", type=str, default='preset/models/dit4sr_f')
parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
parser.add_argument("--process_size", type=int, default=512)
parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
parser.add_argument("--latent_tiled_size", type=int, default=64) 
parser.add_argument("--latent_tiled_overlap", type=int, default=16) 
parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='noise') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
parser.add_argument(
    "--revision",
    type=str,
    default=None,
    required=False,
    help="Revision of pretrained model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--variant",
    type=str,
    default=None,
    help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
)
args = parser.parse_args()

# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def load_dit4sr_pipeline(args, device):

    # Load scheduler, tokenizer and models.
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.transformer_model_name_or_path, subfolder="transformer"
    )
    # controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, subfolder='controlnet')
    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
        )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    transformer.requires_grad_(False)

    # Get the validation pipeline
    validation_pipeline = StableDiffusion3ControlNetPipeline(
        vae=vae, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two, text_encoder_3=text_encoder_three, 
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two, tokenizer_3=tokenizer_three, 
        transformer=transformer, scheduler=scheduler,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    text_encoder_three.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)

    return validation_pipeline

def remove_focus_sentences(text):
    # 使用正则表达式按照 . ? ! 分割，并且保留分隔符本身
    # re.split(pattern, string) 如果 pattern 中带有捕获组()，分隔符也会保留在结果列表中
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense', 'locate', 'position']
    parts = re.split(r'([.?!])', text)
    
    filtered_sentences = []
    i = 0
    while i < len(parts):
        # sentence 可能是句子主体，punctuation 是该句子结尾的标点
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''

        # 组合为完整句子，避免漏掉结尾标点
        full_sentence = sentence + punctuation
        
        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break
        
        # 如果该句子不包含任何禁用词，则保留
        if not skip:
            filtered_sentences.append(full_sentence)
        
        # 跳过已经处理的句子和标点
        i += 2
    
    # 根据需要选择如何重新拼接；这里去掉多余空格并直接拼接
    return "".join(filtered_sentences).strip()


if torch.cuda.device_count() >= 2:
    LLaVA_device = 'cuda:0'
    dit4sr_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    LLaVA_device = 'cuda:0'
    dit4sr_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

llava_agent = LLavaAgent(LLAVA_MODEL_PATH, LLaVA_device, load_8bit=True, load_4bit=False)

# Get the validation pipeline
pipeline = load_dit4sr_pipeline(args, dit4sr_device)

@torch.no_grad()
def process_llava(
    input_image):
    llama_prompt = llava_agent.gen_image_caption([input_image])[0]
    llama_prompt = remove_focus_sentences(llama_prompt)
    return llama_prompt


@torch.no_grad()
def process_sr(
    input_image: Image.Image,
    user_prompt: str,
    positive_prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    scale_factor: int,
    cfg_scale: float,
    seed: int,
    ) -> List[np.ndarray]:
    process_size = 512
    resize_preproc = transforms.Compose([
        transforms.Resize(process_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    seed_everything(seed)
    generator = torch.Generator(device='cuda:0')
    generator.manual_seed(seed)

    validation_prompt = f"{user_prompt} {positive_prompt}"

    ori_width, ori_height = input_image.size
    resize_flag = False

    rscale = scale_factor
    input_image = input_image.resize((int(input_image.size[0] * rscale), int(input_image.size[1] * rscale)))

    if min(input_image.size) < process_size:
        input_image = resize_preproc(input_image)

    input_image = input_image.resize((input_image.size[0] // 8 * 8, input_image.size[1] // 8 * 8))
    width, height = input_image.size
    resize_flag = True  #

    images = []

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    try:
        with torch.autocast("cuda", dtype=weight_dtype, enabled=(args.mixed_precision != "no")):
            image = pipeline(
                prompt=validation_prompt, control_image=input_image, num_inference_steps=num_inference_steps, generator=generator, height=height, width=width,
                guidance_scale=cfg_scale, negative_prompt=negative_prompt, start_point=args.start_point, latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                args=args,
            ).images[0]

        if True:  # alpha<1.0:
            image = adain_color_fix(image, input_image)

        if resize_flag:
            image = image.resize((ori_width * rscale, ori_height * rscale))
    except Exception as e:
        print(e)
        image = Image.new(mode="RGB", size=(512, 512))
    images.append(np.array(image))
    return images



#
Intro= \
"""
## DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution

[Paper](https://arxiv.org/abs/2503.23580)
"""

Prompt= \
"""
First, click \"Run LLAVA\" to generate an initial prompt based on the input image. \\
Then, modify the prompt for higher accuracy. \\
Finally, click \"Run DiT4SR\" to generate the SR result." \
"""

exaple_images = sorted(glob.glob('examples/*.png'))
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(Intro)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil")
            user_prompt = gr.Textbox(label="User Prompt", value="")
            
            with gr.Accordion("Options", open=False):
                positive_prompt = gr.Textbox(label="Positive Prompt", value='Cinematic, perfect without deformations, ultra HD, '
                        'camera, detailed photo, realistic maximum, 32k, Color.')
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value='motion blur, noisy, dotted, pointed, deformed, lowres, chaotic'
                        'CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                        'worst quality, low quality, watermark, signature, jpeg artifacts. '
                )
                cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set a value larger than 1 to enable it!)", minimum=0.1, maximum=10.0, value=7.0, step=0.1)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=20, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=0)
                scale_factor = gr.Number(label="SR Scale", value=4)
            gr.Examples(examples=exaple_images, inputs=[input_image])
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(grid=1)
            with gr.Row():
                run_llava_button = gr.Button(value="Run LLAVA", label="Run LLAVA")
                run_sr_button = gr.Button(value="Run DiT4SR", label="Run DiT4SR")
            gr.Markdown(Prompt)
    
        

    inputs = [
        input_image,
        user_prompt,
        positive_prompt,
        negative_prompt,
        num_inference_steps,
        scale_factor,
        cfg_scale,
        seed,
    ]

    run_llava_button.click(fn=process_llava, inputs=[input_image], outputs=[user_prompt])
    run_sr_button.click(fn=process_sr, inputs=inputs, outputs=[result_gallery])

block.launch()
