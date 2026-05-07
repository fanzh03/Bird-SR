import torch
from PIL import Image
import os
from tqdm import tqdm
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llava.llm_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default='preset/datasets/train_datasets/training_for', help='the dataset you want to tag.') # 
parser.add_argument("--save_dir", type=str, default='preset/datasets/train_datasets/training_for', help='the dataset you want to tag.') # 
parser.add_argument("--stop_num", type=int, default=-1)
parser.add_argument("--start_num", type=int, default=0)
args = parser.parse_args()

def remove_focus_sentences(text):
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense', 'locate', 'position']
    parts = re.split(r'([.?!])', text)
    
    filtered_sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''

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

@torch.no_grad()
def process_llava(
    input_image):
    llama_prompt = llava_agent.gen_image_caption([input_image])[0]
    llama_prompt = remove_focus_sentences(llama_prompt)
    return llama_prompt

def PrintInfo(x):
    if not isinstance(x,list):
        x=[x]
    for i in x:
        print('shape : {} ; dtype : {} ; max : {} ; min : {}'.format(i.shape,i.dtype,i.max(),i.min())  )

img_folder = args.img_dir
prompt_save_folder = args.save_dir
os.makedirs(prompt_save_folder, exist_ok=True)
img_name_list = os.listdir(img_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device, load_8bit=True, load_4bit=False)

for img_name in tqdm(img_name_list):
    if os.path.exists(os.path.join(prompt_save_folder, img_name.replace('png', 'txt'))):
        continue
    img_path = os.path.join(img_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    prompt = process_llava(img)
    with open(os.path.join(prompt_save_folder, img_name.replace('png', 'txt')), 'w', encoding="utf-8") as f:
        f.write(prompt)