#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import sys

project_root = "/path/to/DiT4SR-main"
sys.path.insert(0, project_root)

import argparse
import copy
import logging
import math
import glob
import shutil
from pathlib import Path
import re
from PIL import Image
import random

import accelerate
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from pipelines.pipeline_dit4sr import StableDiffusion3ControlNetPipeline
from torchvision import transforms

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    # StableDiffusion3ControlNetPipeline,
)
from model_dit4sr.transformer_sd3 import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory, \
    cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.torch_utils import randn_tensor

from diffusers.image_processor import VaeImageProcessor
from torchvision.utils import save_image
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from dataloaders.paired_dataset_sd3_latent import PairedCaptionDataset
import pyiqa
from dinov2.hub.backbones import _make_dinov2_model

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='/PATH/DiT4SR-main/preset/models/stable-diffusion-3.5-medium',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_model_name_or_path",
        type=str,
        default='/PATH/DiT4SR-main/preset/models/dit4sr_q',
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
             " If not specified controlnet weights are initialized from unet.",
    )
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/PATH/Data/SR/DIV8K/experiments/dit4sr",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
             "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=0.001, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='NOTHING',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument("--root_folders", type=str, default='/PATH/Data/SR/DIV8K/train_LR')
    parser.add_argument("--null_text_ratio", type=float, default=0.2)
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["control"])

    # loss settings
    parser.add_argument("--lpips_loss", type=int, default=1)
    parser.add_argument("--lpips_ckpt", type=str,
                        default='/PATH/DiT4SR-main/preset/IQAWeights/lpips/vgg.pth',
                        help="Need lpips_ckpt !!!!")
    parser.add_argument("--lpips_loss_weight", type=float, default=1)

    parser.add_argument("--musiq_loss", type=int, default=0)
    parser.add_argument("--musiq_ckpt", type=str,
                        default='/PATH/DiT4SR-main/preset/IQAWeights/musiq/musiq_koniq_ckpt-e95806b9.pth',
                        help="Need musiq_ckpt !!!!")
    parser.add_argument("--musiq_loss_weight", type=float, default=0.00001)

    parser.add_argument("--clipiqa_loss", type=int, default=1)
    parser.add_argument("--clipiqa_ckpt", type=str,
                        default='/PATH/DiT4SR-main/preset/IQAWeights/cilpiqa/RN50.pt',
                        help="Need clipiqa_ckpt !!!!")
    parser.add_argument("--clipiqa_loss_weight", type=float, default=0.00001)

    parser.add_argument("--liqe_loss", type=int, default=0)
    parser.add_argument("--liqe_ckpt", type=str,
                        default='/PATH/DiT4SR-main/preset/IQAWeights/liqe/liqe_koniq.pt',
                        help="Need liqe_ckpt !!!!")
    parser.add_argument("--liqe_loss_weight", type=float, default=0.00001)

    parser.add_argument("--maniqa_loss", type=int, default=1)
    parser.add_argument("--maniqa_ckpt", type=str,
                        default='/PATH/DiT4SR-main/preset/IQAWeights/maniqa/ckpt_koniq10k.pt',
                        help="Need maniqa_ckpt !!!!")
    parser.add_argument("--maniqa_loss_weight", type=float, default=0.00001)

    parser.add_argument("--dino_loss", type=int, default=1)
    parser.add_argument("--dino_ckpt", type=str,
                        default='/PATH/DiT4SR-main/preset/models/DINO/dinov2_vitl14_reg4_pretrain.pth',
                        help="Need dino_kl_ckpt !!!!")
    # /PATH/DiT4SR-main/preset/models/facebook/dinov3-vitb16-pretrain-lvd1689m
    parser.add_argument("--dino_type", type=str, default='vit_large', choices=["vit_small", "vit_base", "vit_large"],
                        help="Need dino_type !!!!")
    parser.add_argument("--dino_loss_weight", type=float, default=0.0001)
    parser.add_argument("--dino_loss_type", type=str, default='MSE', choices=["Cosine", "MSE"],
                        help="Need dino_loss_type Cosine or MSE !!!!")

    parser.add_argument("--validation_image_dir", type=str,
                        default="/PATH/Data/SR/RealLR200")
    parser.add_argument("--validation_prompt_dir", type=str,
                        default="/PATH/Data/SR/llavaCaptionRealLR200/txt")
    parser.add_argument("--negative_prompt", type=str, default='motion blur, noisy, dotted, bokeh, pointed, '
                                                               'CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                                                               'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                                                               'deformed, lowres, chaotic')
    parser.add_argument("--process_size", type=int, default=512)
    # parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    # parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=64)
    parser.add_argument("--latent_tiled_overlap", type=int, default=24)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'],
                        default='noise')  # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point.
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--do_grad_inference_steps", type=int, default=40)
    parser.add_argument("--lr_only_folder", type=str, default='')
    parser.add_argument("--train_eta_gamma", type=float, default=8.0)
    parser.add_argument("--train_total_loss_weight", type=float, default=1.0)  # 1.0

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def update_ema(target_params, current_params, rate=0.999):
    for targ, cur in zip(target_params, current_params):
        targ.detach().mul_(rate).add_(cur, alpha=1 - rate)


# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def compute_lpips_loss(pred_image, target_image, lpips_loss):
    pred = pred_image * 2 - 1  # [0,1] -> [-1,1]
    target = target_image * 2 - 1
    pred = pred_image.clamp(-1, 1)
    target = target_image.clamp(-1, 1)
    loss = lpips_loss(pred, target).to(pred.dtype)

    return loss


def compute_reward_loss(pred_image, reward_loss, target_image=None, reward_type="clipiqa"):
    reward_score_pred = reward_loss(pred_image.float())  # [0,1]
    if target_image is not None:
        reward_score_target = reward_loss(target_image.float())
    if target_image is not None:
        score = F.relu(reward_score_target.mean() - reward_score_pred.mean() + 0.04)
    else:
        score = F.relu(-reward_score_pred.mean() + 1)
    return score


def compute_dino_loss(pred_image, old_model_pred, dino_loss, loss_type="MSE"):
    mean = torch.tensor([0.485, 0.456, 0.406], device=pred_image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=pred_image.device).view(1, 3, 1, 1)

    # [0,1]
    def preprocess(img):
        img_norm = (img - mean) / std
        return F.interpolate(img_norm, size=(392, 392), mode="bilinear", align_corners=False)

    pred_in = preprocess(pred_image)
    old_in = preprocess(old_model_pred)

    pred_feat = dino_loss.forward_features(pred_in)['x_norm_patchtokens']  # [B, N, C]

    old_feat = dino_loss.forward_features(old_in)['x_norm_patchtokens']

    if loss_type == "Cosine":

        pred_norm = F.normalize(pred_feat, dim=-1)
        old_norm = F.normalize(old_feat, dim=-1)

        cos_sim = (pred_norm * old_norm).sum(dim=-1)  # [B, N]
        loss = 1.0 - cos_sim.mean()
    elif loss_type == "MSE":

        loss = F.mse_loss(pred_feat, old_feat)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    return loss



def load_validation_pairs(image_dir, prompt_dir):

    image_ext = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    images = []
    for ext in image_ext:
        images.extend(glob.glob(os.path.join(image_dir, ext)))


    prompt_files = glob.glob(os.path.join(prompt_dir, "*.txt"))

    image_dict = {
        os.path.splitext(os.path.basename(img))[0]: img
        for img in images
    }
    prompt_dict = {
        os.path.splitext(os.path.basename(txt))[0]: txt
        for txt in prompt_files
    }

    common_keys = sorted(list(set(image_dict.keys()) & set(prompt_dict.keys())))

    if len(common_keys) == 0:
        raise ValueError("no matched image-txt pairs!!!")

    validation_images = [image_dict[k] for k in common_keys]
    validation_prompts = [prompt_dict[k] for k in common_keys]

    return validation_images, validation_prompts


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


def load_dit4sr_pipeline(transformer_target, args, accelerator, weight_dtype):
    # from model_dit4sr.transformer_sd3 import SD3Transformer2DModel

    # Load scheduler, tokenizer and models.

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )

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
    transformer_target.requires_grad_(False)

    # Get the validation pipeline
    validation_pipeline = StableDiffusion3ControlNetPipeline(
        vae=vae, text_encoder=text_encoder_one, text_encoder_2=text_encoder_two, text_encoder_3=text_encoder_three,
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two, tokenizer_3=tokenizer_three,
        transformer=transformer_target, scheduler=scheduler,
    )


    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer_target.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline


def remove_focus_sentences(text):
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense',
                        'locate', 'position']
    parts = re.split(r'([.?!])', text)

    filtered_sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i]
        punctuation = parts[i + 1] if (i + 1 < len(parts)) else ''

        full_sentence = sentence + punctuation

        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break

        if not skip:
            filtered_sentences.append(full_sentence)

        i += 2
    return "".join(filtered_sentences).strip()


def log_validation(transformer_target, args, accelerator, weight_dtype, global_step, validation_image_list,
                   validation_prompt_list, iqa_model=None):
    pipeline = load_dit4sr_pipeline(transformer_target, args, accelerator, weight_dtype)
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        score = []
        if iqa_model is not None:
            iqa_model = iqa_model.to(accelerator.device)
            iqa_model.eval()

        pairs = list(zip(validation_image_list, validation_prompt_list))
        random.shuffle(pairs)

        pairs = pairs[:-1]
        for image_idx, (image_name, prompt_name) in enumerate(pairs):
            print(f'================== step {global_step} validate - process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")
            with open(prompt_name, 'r') as f:
                validation_prompt = f.read()
            # print(f'before remove_focus_sentences {validation_prompt}')
            validation_prompt = remove_focus_sentences(validation_prompt)
            negative_prompt = args.negative_prompt  # dirty, messy, low quality, frames, deformed,
            # print(f'remove_focus_sentences {validation_prompt}')
            # print(f'negative_prompt {negative_prompt}')

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale
            if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
                scale = (args.process_size // rscale) / min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale * ori_width), int(scale * ori_height)), Image.BICUBIC)

                validation_image = tmp_image
                resize_flag = True

            validation_image = validation_image.resize(
                (validation_image.size[0] * rscale, validation_image.size[1] * rscale), Image.BICUBIC)
            validation_image = validation_image.resize(
                (validation_image.size[0] // 8 * 8, validation_image.size[1] // 8 * 8), Image.BICUBIC)
            width, height = validation_image.size
            resize_flag = True  #

            # print(f'input size: {height}x{width}')

            for sample_idx in range(args.sample_times):
                os.makedirs(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)

            for sample_idx in range(args.sample_times):
                with torch.autocast("cuda"):
                    # start_time = time.time()
                    image = pipeline(
                        prompt=validation_prompt, control_image=validation_image,
                        num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                        guidance_scale=args.guidance_scale, negative_prompt=negative_prompt,
                        start_point=args.start_point, latent_tiled_size=args.latent_tiled_size,
                        latent_tiled_overlap=args.latent_tiled_overlap,
                        args=args,
                    ).images[0]
                    # end_time = time.time()
                    # print(f'inference time: {end_time-start_time:.2f}s')

                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)

                if resize_flag:
                    image = image.resize((ori_width * rscale, ori_height * rscale), Image.BICUBIC)

                name, ext = os.path.splitext(os.path.basename(image_name))
                image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
                print(
                    f'================== process {image_idx} imgs... save to: {args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png ===================')

                # --- IQA (CLIPIQA / MUSIQ / etc.) ---
                pred_tensor = transforms.ToTensor()(image).unsqueeze(0).to(accelerator.device)
                if iqa_model is not None:
                    with torch.no_grad():
                        # 如果是 pyiqa，需要 BxCxHxW tensor
                        iqa_val = iqa_model(pred_tensor).item()

                        print(f'================== iqa_val {iqa_val} - process {image_idx} imgs... ===================')
                    score.append(iqa_val)
        # -------- final score --------
        if len(score) > 0:
            avg_score = sum(score) / len(score)
            print(
                f'================== step {global_step} validate all process {image_idx + 1} images... ===================')
            print(
                f'================== average iqa_score of validate all process {image_idx + 1} images: {avg_score} ===================')
        else:
            print("No IQA scores computed.")


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    noise_scheduler_copy_unpair = copy.deepcopy(noise_scheduler)
    if args.dino_loss is not None:
        noise_scheduler_copy_unpair_old = copy.deepcopy(noise_scheduler)

    if args.transformer_model_name_or_path is not None:
        # for policy
        transformer = SD3Transformer2DModel.from_pretrained_local(
            args.transformer_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
        # for ema
        transformer_target = SD3Transformer2DModel.from_pretrained_local(
            args.transformer_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
        # for reference
        transformer_old = SD3Transformer2DModel.from_pretrained_local(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
        # for pixel image
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
        )
    else:
        transformer = SD3Transformer2DModel.from_pretrained_local(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    transformer_target.requires_grad_(False)
    transformer_old.requires_grad_(False)

    vae_scale_factor = (
        2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
    )
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # release the cross-attention part in the unet.
    for name, params in transformer.named_parameters():
        # if name.endswith(tuple(args.trainable_modules)):
        if any(trainable_modules in name for trainable_modules in tuple(args.trainable_modules)):
            print(f'{name} in <transformer> will be optimized.')
            # for params in module.parameters():
            params.requires_grad = True

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unwrap_model(transformer).save_pretrained(os.path.join(output_dir, "transformer"))
                unwrap_model(transformer_target).save_pretrained(os.path.join(output_dir, "transformer_target"))

                while len(weights) > 0:
                    weights.pop()
                while len(models) > 0:
                    models.pop()



        def load_model_hook(models, input_dir):
            # while len(models) > 0:
            # pop models so that they are not loaded again

            # 1. transformer
            load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
            transformer_instance = [m for m in models if m is unwrap_model(transformer)][0]
            transformer_instance.register_to_config(**load_model.config)
            transformer_instance.load_state_dict(load_model.state_dict())
            del load_model

            # 2. transformer_target EMA
            load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer_target")
            transformer_target_instance = [m for m in models if m is unwrap_model(transformer_target)][0]
            transformer_target_instance.register_to_config(**load_model.config)
            transformer_target_instance.load_state_dict(load_model.state_dict())
            del load_model

            # 3.transformer_old reference
            load_model = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,
                                                               subfolder="transformer")
            transformer_old_instance = [m for m in models if m is unwrap_model(transformer_old)][0]
            transformer_old_instance.register_to_config(**load_model.config)
            transformer_old_instance.load_state_dict(load_model.state_dict())
            del load_model

            while len(models) > 0:
                models.pop()


        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    # params_to_optimize = controlnet.parameters()
    # params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    # params_to_optimize = [transformer_parameters_with_lr]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    ################# set loss models from preteained models #####################
    clipiqa_loss, lpips_loss, dino_loss, maniqa_loss = None, None, None, None
    if args.clipiqa_loss:
        clipiqa_ckpt = args.clipiqa_ckpt
        # clipiqa_loss = pyiqa.create_metric('clipiqa', as_loss=True, backbone=clipiqa_ckpt).to(accelerator.device)
        clipiqa_loss = pyiqa.create_metric('clipiqa', backbone=clipiqa_ckpt).to(accelerator.device)
        clipiqa_loss.requires_grad_(False)
        clipiqa_loss.eval()
        logger.info(f"***** Begin compiling clipiqa Metric=>{args.clipiqa_ckpt}, Successfully!!! *****")

    if args.lpips_loss:
        lpips_ckpt = args.lpips_ckpt
        # # -------------------------------------------------
        # # 3. 构建 LPIPS（此时不会触发下载）
        # # -------------------------------------------------
        import lpips
        lpips_loss = lpips.LPIPS(net="vgg", pretrained=False, pnet_rand=False)  # 随机初始化权重

        state_dict = torch.load(lpips_ckpt, map_location="cpu")
        lpips_loss.load_state_dict(state_dict, strict=False)  # strict=False 允许权重不完全匹配，这可能导致 trunk 部分没有被正确赋值。

        lpips_loss.eval()
        lpips_loss.requires_grad_(False)
        lpips_loss = lpips_loss.to(accelerator.device)
        logger.info(f"***** Begin compiling lpips Metric=>{args.lpips_ckpt}, Successfully!!!  *****")

    if args.dino_loss:
        dino_ckpt = args.dino_ckpt
        dino_loss = _make_dinov2_model(arch_name=args.dino_type, pretrained=False, num_register_tokens=4)
        state_dict = torch.load(dino_ckpt, map_location=accelerator.device)
        dino_loss.load_state_dict(state_dict)
        dino_loss.to(accelerator.device)
        dino_loss.requires_grad_(False)
        dino_loss.eval()
        logger.info(f"***** Begin compiling DINOv2 Metric=>{args.dino_ckpt},{args.dino_type},  Successfully!!! *****")

    if args.maniqa_loss:
        maniqa_ckpt = args.maniqa_ckpt
        # maniqa_loss = pyiqa.create_metric('maniqa', as_loss=True, pretrained_model_path=maniqa_ckpt).to(accelerator.device)
        maniqa_loss = pyiqa.create_metric('maniqa', pretrained_model_path=maniqa_ckpt).to(accelerator.device)
        maniqa_loss.requires_grad_(False)
        maniqa_loss.eval()
        logger.info(f"***** Begin compiling maniqa Metric=>{args.maniqa_ckpt}, Successfully!!! *****")
    if args.liqe_loss:
        liqe_ckpt = args.liqe_ckpt
        liqe_loss = pyiqa.create_metric('liqe', backbone="preset/IQAWeights/cilpiqa/ViT-B-32.pt",
                                        pretrained_model_path=liqe_ckpt).to(
            accelerator.device)  # liqe_ckpt= preset/IQAWeights/liqe/liqe_koniq.pt
        liqe_loss.requires_grad_(False)
        liqe_loss.eval()
        logger.info(f"***** Begin compiling liqe Metric=>{args.liqe_ckpt}, Successfully!!! *****")

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer_old.to(accelerator.device, dtype=weight_dtype)
    transformer_target.to(accelerator.device, dtype=weight_dtype)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    train_dataset = PairedCaptionDataset(root_folder=args.root_folders,
                                         lr_only_folder=args.lr_only_folder,
                                         null_text_ratio=args.null_text_ratio,
                                         )

    def compute_text_embeddings(batch, text_encoders, tokenizers):
        with torch.no_grad():
            prompt = batch["prompts"]
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    ###################################  #########################################
    validation_image, validation_prompt = load_validation_pairs(
        args.validation_image_dir,
        args.validation_prompt_dir
    )
    logger.info(f"############################################################################")
    logger.info(f"validation_prompt ={validation_prompt}")
    logger.info(f"validation_image ={validation_image}")
    logger.info(f"############################################################################")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    vae, transformer_old, transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, transformer_old, transformer, optimizer, train_dataloader, lr_scheduler
    )
    # transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     transformer, optimizer, train_dataloader, lr_scheduler
    # )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        tracker_config = {
            k: (str(v) if isinstance(v, (list, dict)) else v)
            for k, v in tracker_config.items()
        }

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            initial_global_step = 0

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    free_memory()

    # Check if debug mode is enabled
    debug_dimensions = os.environ.get("DEBUG_DIMENSIONS", "0") == "1"

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                if 'is_unpair' in batch:
                    is_unpair_mask = batch['is_unpair'].bool()  # .to(device=accelerator.device)
                else:
                    is_unpair_mask = torch.zeros_like(batch['conditioning_pixel_values'], dtype=torch.bool)
                loss_dict = {}
                ###################                    reward-guided  Loss                          #########################
                # Change these lines:
                loss_rl = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                loss_rl_unpair = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                if (~is_unpair_mask).any():

                    # Convert images to latent space
                    model_input = batch["pixel_values"][~is_unpair_mask].to(dtype=weight_dtype)

                    # Debug: Print input dimensions
                    if debug_dimensions and step == 0:
                        logger.info("=" * 80)
                        logger.info("DEBUG: Input Data Dimensions Tracking")
                        logger.info("=" * 80)
                        logger.info(f"Step {step}, Epoch {epoch}")
                        logger.info(f"  batch['pixel_values'] shape: {batch['pixel_values'].shape}")
                        logger.info(f"  batch['pixel_values'] dtype: {batch['pixel_values'].dtype}")

                    # controlnet(s) inference
                    controlnet_image = batch["conditioning_pixel_values"][~is_unpair_mask].to(dtype=weight_dtype)

                    # Debug: Print controlnet image dimensions
                    if debug_dimensions and step == 0:
                        logger.info(f"  model_input (after to dtype) shape: {model_input.shape}")
                        logger.info(f"  model_input dtype: {model_input.dtype}")
                        logger.info(
                            f"  batch['conditioning_pixel_values'] shape: {batch['conditioning_pixel_values'].shape}")
                        logger.info(f"  controlnet_image (after to dtype) shape: {controlnet_image.shape}")
                        logger.info(f"  controlnet_image dtype: {controlnet_image.dtype}")


                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    # Debug: Print noise dimensions
                    if debug_dimensions and step == 0:
                        logger.info(f"  noise shape: {noise.shape}")
                        logger.info(f"  batch size (bsz): {bsz}")

                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                    # Add noise according to flow matching.
                    # zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    # Debug: Print noisy input dimensions
                    if debug_dimensions and step == 0:
                        logger.info(f"  timesteps shape: {timesteps.shape}")
                        logger.info(f"  timesteps values (sample): {timesteps[:min(5, len(timesteps))]}")
                        logger.info(f"  sigmas shape: {sigmas.shape}")
                        logger.info(f"  sigmas values (sample): {sigmas.flatten()[:min(5, len(sigmas.flatten()))]}")
                        logger.info(f"  noisy_model_input shape: {noisy_model_input.shape}")
                        logger.info(f"  noisy_model_input dtype: {noisy_model_input.dtype}")
                    # input_model_input = torch.cat([noisy_model_input, controlnet_image], dim = 1)

                    # Get the text embedding for conditioning
                    # prompts = compute_text_embeddings(batch, text_encoders, tokenizers)
                    prompt_embeds = batch["prompt_embeds"][~is_unpair_mask].to(dtype=model_input.dtype)
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"][~is_unpair_mask].to(dtype=model_input.dtype)

                    # Debug: Print text embedding dimensions
                    if debug_dimensions and step == 0:
                        logger.info(f"  batch['prompt_embeds'] shape: {batch['prompt_embeds'].shape}")
                        logger.info(f"  prompt_embeds (after to dtype) shape: {prompt_embeds.shape}")
                        logger.info(f"  prompt_embeds dtype: {prompt_embeds.dtype}")
                        logger.info(f"  batch['pooled_prompt_embeds'] shape: {batch['pooled_prompt_embeds'].shape}")
                        logger.info(f"  pooled_prompt_embeds (after to dtype) shape: {pooled_prompt_embeds.shape}")
                        logger.info(f"  pooled_prompt_embeds dtype: {pooled_prompt_embeds.dtype}")
                        logger.info("-" * 80)
                        logger.info("DEBUG: Transformer Input Dimensions")
                        logger.info("-" * 80)
                        logger.info(f"  hidden_states (noisy_model_input) shape: {noisy_model_input.shape}")
                        logger.info(f"  controlnet_image shape: {controlnet_image.shape}")
                        logger.info(f"  timestep shape: {timesteps.shape}")
                        logger.info(f"  encoder_hidden_states (prompt_embeds) shape: {prompt_embeds.shape}")
                        logger.info(f"  pooled_projections (pooled_prompt_embeds) shape: {pooled_prompt_embeds.shape}")

                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        controlnet_image=controlnet_image,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

                    # Debug: Print model output dimensions
                    if debug_dimensions and step == 0:
                        logger.info("-" * 80)
                        logger.info("DEBUG: Transformer Output Dimensions")
                        logger.info("-" * 80)
                        logger.info(f"  model_pred shape: {model_pred.shape}")
                        logger.info(f"  model_pred dtype: {model_pred.dtype}")

                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    # Preconditioning of the model outputs.
                    if args.precondition_outputs:
                        model_pred = model_pred * (-sigmas) + noisy_model_input

                    # Debug: Print preconditioned output dimensions
                    if debug_dimensions and step == 0:
                        logger.info(f"  model_pred (after preconditioning) shape: {model_pred.shape}")

                    # these weighting schemes use a uniform timestep sampling
                    # and instead post-weight the loss
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                    # flow matching loss
                    if args.precondition_outputs:
                        target = model_input
                    else:
                        target = noise - model_input

                    # Debug: Print loss computation dimensions
                    if debug_dimensions and step == 0:
                        logger.info("-" * 80)
                        logger.info("DEBUG: Loss Computation Dimensions")
                        logger.info("-" * 80)
                        logger.info(f"  weighting shape: {weighting.shape}")
                        logger.info(f"  target shape: {target.shape}")
                        logger.info(f"  target dtype: {target.dtype}")
                        logger.info(f"  model_pred (for loss) shape: {model_pred.shape}")
                        logger.info(f"  (model_pred - target) shape: {(model_pred.float() - target.float()).shape}")
                        logger.info("=" * 80)

                    ###################         Add VAE.decoder for pixel image            #########################
                    model_pred = (model_pred / vae.config.scaling_factor) + vae.config.shift_factor
                    model_pred = model_pred.to(dtype=weight_dtype)
                    pred_image = vae.decode(model_pred).sample  # [-1,1]
                    # ->[0,1]
                    pred_image = (pred_image + 1) / 2
                    pred_image = pred_image.clamp(0, 1)

                    target = (target / vae.config.scaling_factor) + vae.config.shift_factor
                    target = target.to(dtype=weight_dtype)
                    target_image = vae.decode(target).sample  # [-1,1]
                    # ->[0,1]
                    target_image = (target_image + 1) / 2
                    target_image = target_image.clamp(0, 1)

                    save_image(pred_image, f'{args.output_dir}/pred_image.png')
                    save_image(target_image, f'{args.output_dir}/target_image.png')
                    # ----dynamic n_t ---
                    gamma = args.train_eta_gamma  # [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
                    total_weight = args.train_total_loss_weight  # 1.0
                    # n_t
                    eta_t_vec = (timesteps / noise_scheduler_copy.config.num_train_timesteps).clamp(0, 1) ** gamma
                    eta_t_vec = eta_t_vec.to(pred_image.device, dtype=weight_dtype)  #
                    w_lpips_vec = total_weight * eta_t_vec
                    w_reward_vec = total_weight * (1 - eta_t_vec)
                    ###################         Compute Reward Loss from pixel image       ##########################
                    if args.lpips_loss and lpips_loss is not None:
                        lpips_loss_weight = args.lpips_loss_weight
                        loss_lpips = compute_lpips_loss(pred_image, target_image, lpips_loss) * lpips_loss_weight
                        loss_rl = loss_rl + loss_lpips * w_lpips_vec
                        logger.info(f'"loss_lpips":{loss_lpips.detach().mean().item()}')
                    if args.clipiqa_loss and clipiqa_loss is not None:
                        clipiqa_loss_weight = args.clipiqa_loss_weight
                        loss_clipiqa = compute_reward_loss(pred_image, clipiqa_loss, target_image,
                                                           reward_type="clipiqa") * clipiqa_loss_weight
                        loss_rl = loss_rl + loss_clipiqa * w_reward_vec
                        logger.info(f'"loss_clipiqa":{loss_clipiqa.detach().mean().item()}')
                    if args.maniqa_loss and maniqa_loss is not None:
                        maniqa_loss_weight = args.maniqa_loss_weight
                        loss_maniqa = compute_reward_loss(pred_image, maniqa_loss, target_image,
                                                          reward_type="maniqa") * maniqa_loss_weight
                        loss_rl = loss_rl + loss_maniqa * w_reward_vec
                        logger.info(f'"loss_maniqa":{loss_maniqa.detach().mean().item()}')

                    if debug_dimensions and step == 0:
                        logger.info(f"  loss value: {loss_rl.item()}")
                        logger.info("=" * 80)

                    accelerator.backward(loss_rl.mean())
                if (is_unpair_mask).any():
                    generator = torch.Generator(device=accelerator.device)
                    if args.seed is not None:
                        generator.manual_seed(args.seed)

                    inference_timesteps = args.num_inference_steps
                    do_grad = args.do_grad_inference_steps

                    latents = randn_tensor(batch["pixel_values"][is_unpair_mask].shape, generator=generator,
                                           device=accelerator.device, dtype=weight_dtype)
                    if args.dino_loss and dino_loss is not None:
                        latents_old = randn_tensor(batch["pixel_values"][is_unpair_mask].shape, generator=generator,
                                                   device=accelerator.device, dtype=weight_dtype)
                    noise_scheduler_copy_unpair.set_timesteps(
                        inference_timesteps,
                        device=latents.device
                    )
                    if args.dino_loss and dino_loss is not None:
                        noise_scheduler_copy_unpair_old.set_timesteps(
                            inference_timesteps,
                            device=latents.device
                        )
                    inf_timesteps = noise_scheduler_copy_unpair.timesteps
                    for i, t in enumerate(inf_timesteps):  # [:-1]
                        latent_model_input = latents
                        if args.dino_loss and dino_loss is not None:
                            latent_model_input_old = latents_old
                        timestep = t.expand(latent_model_input.shape[0])
                        controlnet_image = batch["conditioning_pixel_values"][is_unpair_mask].to(dtype=weight_dtype)
                        prompt_embeds = batch["prompt_embeds"][is_unpair_mask].to(dtype=latent_model_input.dtype)
                        pooled_prompt_embeds = batch["pooled_prompt_embeds"][is_unpair_mask].to(
                            dtype=latent_model_input.dtype)
                        if i < (do_grad - 1):
                            ctx = torch.no_grad()
                        else:
                            ctx = torch.enable_grad()
                        with ctx:
                            noise_pred = transformer(
                                hidden_states=latent_model_input,
                                controlnet_image=controlnet_image,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                return_dict=False,
                            )[0]

                        if args.dino_loss and dino_loss is not None:
                            old_model_pred = transformer_old(
                                hidden_states=latent_model_input_old,
                                controlnet_image=controlnet_image,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                return_dict=False,
                            )[0]

                        latents_dtype = latents.dtype
                        latents = noise_scheduler_copy_unpair.step(noise_pred, t, latents, return_dict=False)[0]
                        if args.dino_loss and dino_loss is not None:
                            latents_old = \
                            noise_scheduler_copy_unpair_old.step(old_model_pred, t, latents_old, return_dict=False)[0]
                        if latents.dtype != latents_dtype:
                            if torch.backends.mps.is_available():
                                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                                latents = latents.to(latents_dtype)

                    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
                    latents = latents.to(dtype=weight_dtype)
                    image = vae.decode(latents, return_dict=False)[0]
                    image = (image + 1) / 2
                    image = image.clamp(0, 1)

                    if args.dino_loss and dino_loss is not None:
                        latents_old = (latents_old / vae.config.scaling_factor) + vae.config.shift_factor
                        latents_old = latents_old.to(dtype=weight_dtype)
                        old_image = vae.decode(latents_old, return_dict=False)[0]
                        old_image = (old_image + 1) / 2
                        old_image = old_image.clamp(0, 1)

                    ###################         Compute Reward Loss from pixel image       ##########################
                    if args.clipiqa_loss and clipiqa_loss is not None:
                        clipiqa_loss_weight = args.clipiqa_loss_weight
                        loss_clipiqa_unpair = compute_reward_loss(image, clipiqa_loss,
                                                                  reward_type="clipiqa") * clipiqa_loss_weight
                        loss_rl_unpair = loss_rl_unpair + loss_clipiqa_unpair
                        logger.info(f'"loss_clipiqa_unpair":{loss_clipiqa_unpair.detach().mean().item()}')
                    if args.dino_loss and dino_loss is not None:
                        dino_loss_weight = args.dino_loss_weight
                        loss_dino = compute_dino_loss(image, old_image, dino_loss,
                                                      args.dino_loss_type) * dino_loss_weight
                        loss_rl_unpair = loss_rl_unpair + loss_dino
                        logger.info(f'"loss_dino":{loss_dino.detach().mean().item()}')
                    if args.maniqa_loss and maniqa_loss is not None:
                        maniqa_loss_weight = args.maniqa_loss_weight
                        loss_maniqa_unpair = compute_reward_loss(image, maniqa_loss,
                                                                 reward_type="maniqa") * maniqa_loss_weight
                        loss_rl_unpair = loss_rl_unpair + loss_maniqa_unpair
                        logger.info(f'"loss_maniqa_unpair":{loss_maniqa_unpair.detach().mean().item()}')
                    accelerator.backward(loss_rl_unpair.mean())
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                update_ema(transformer_target.parameters(), transformer.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        logger.info(f"Begin validates for {args.validation_image_dir}")
                        image_logs = log_validation(
                            transformer_target,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            validation_image,
                            validation_prompt,
                            clipiqa_loss,
                        )
            # logger.info(f"loss_rl.requires_grad:{loss_rl.requires_grad}")
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            unpair_num = is_unpair_mask.sum().item()
            if unpair_num == 0:
                loss_rl_unpair = torch.zeros(1, device=accelerator.device)
            logs = {"unpair_num": unpair_num, "loss_rl": loss_rl.detach().mean().item(),
                    "loss_rl_unpair": loss_rl_unpair.detach().mean().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
