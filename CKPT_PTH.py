import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

LLAVA_CLIP_PATH = os.path.join(_REPO_ROOT, 'llava_ckpt/clip-vit-large-patch14-336')
LLAVA_MODEL_PATH = os.path.join(_REPO_ROOT, 'llava_ckpt/llava-v1.5-13b')
