import glob
import os
import random

import torch
from torchvision import transforms
from torch.utils import data as data


class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folder=None,
            lr_only_folder=None,
            null_text_ratio=0.2,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        # self.tag_path_list = []
        self.prompt_embeds_path_list = []
        self.pooled_prompt_embeds_path_list = []
        self.unpair_list = []


        # for root_folder in root_folders:
        lr_path = root_folder +'/latent_lr'
        # tag_path = root_folder +'/tag'
        gt_path = root_folder +'/latent_hr'
        prompt_embeds_path = root_folder + '/prompt_embeds'
        pooled_prompt_embeds_path = root_folder + '/pooled_prompt_embeds'

        # Check if directories exist
        dirs_to_check = {
            'latent_lr': lr_path,
            'latent_hr': gt_path,
            'prompt_embeds': prompt_embeds_path,
            'pooled_prompt_embeds': pooled_prompt_embeds_path
        }
        
        missing_dirs = []
        for name, path in dirs_to_check.items():
            if not os.path.exists(path):
                missing_dirs.append(f"{name}: {path}")
        
        self.lr_list += glob.glob(os.path.join(lr_path, '*.pt'))
        self.gt_list += glob.glob(os.path.join(gt_path, '*.pt'))
        # self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))
        self.prompt_embeds_path_list += glob.glob(os.path.join(prompt_embeds_path, '*.pt'))
        self.pooled_prompt_embeds_path_list += glob.glob(os.path.join(pooled_prompt_embeds_path, '*.pt'))
        
        # Diagnostic information if directories are missing or empty
        if missing_dirs:
            raise ValueError(
                f"Missing directories:\n" + "\n".join(missing_dirs) + "\n" +
                f"Please check your root_folder: {root_folder}"
            )
        
        # Check for empty directories and provide helpful diagnostics
        file_counts = {
            'latent_lr': len(self.lr_list),
            'latent_hr': len(self.gt_list),
            'prompt_embeds': len(self.prompt_embeds_path_list),
            'pooled_prompt_embeds': len(self.pooled_prompt_embeds_path_list)
        }
        
        if file_counts['latent_lr'] == 0:
            # Check if directory exists and list what's in it
            if os.path.exists(lr_path):
                all_files = os.listdir(lr_path)
                sample_files = all_files[:5] if len(all_files) > 0 else []
                raise ValueError(
                    f"No .pt files found in {lr_path}. "
                    f"Directory exists but contains {len(all_files)} files. "
                    f"Sample files: {sample_files}. "
                    f"Please check if files have .pt extension or if directory structure is correct."
                )
            else:
                raise ValueError(
                    f"Directory does not exist: {lr_path}. "
                    f"Please check your root_folder: {root_folder}"
                )

        self.null_pooled_prompt_embeds_path = os.path.join(pooled_prompt_embeds_path, 'NULL_pooled_prompt_embeds.pt')
        self.null_prompt_embeds_path = os.path.join(prompt_embeds_path, 'NULL_prompt_embeds.pt')

        if self.null_prompt_embeds_path in self.prompt_embeds_path_list:
            self.prompt_embeds_path_list.remove(self.null_prompt_embeds_path)
        if self.null_pooled_prompt_embeds_path in self.pooled_prompt_embeds_path_list:
            self.pooled_prompt_embeds_path_list.remove(self.null_pooled_prompt_embeds_path)

        # Check if NULL files exist
        null_files_exist = os.path.exists(self.null_prompt_embeds_path) and os.path.exists(self.null_pooled_prompt_embeds_path)
        
        if not null_files_exist and self.null_text_ratio > 0:
            import warnings
            warnings.warn(
                f"NULL prompt files not found ({self.null_prompt_embeds_path}, {self.null_pooled_prompt_embeds_path}). "
                f"Disabling null_text_ratio (was {self.null_text_ratio}). "
                f"If you want to use null text, please create these files.",
                UserWarning
            )
            self.null_text_ratio = 0
        
        # Pre-load NULL files if they exist to avoid repeated file I/O
        self.null_prompt_embeds = None
        self.null_pooled_prompt_embeds = None
        if null_files_exist:
            self.null_prompt_embeds = torch.load(self.null_prompt_embeds_path)
            self.null_pooled_prompt_embeds = torch.load(self.null_pooled_prompt_embeds_path)

        # Sort all lists to ensure consistent ordering
        self.lr_list.sort()
        self.gt_list.sort()
        self.prompt_embeds_path_list.sort()
        self.pooled_prompt_embeds_path_list.sort()

        # Create basename mappings for all file lists
        lr_basenames = {os.path.basename(p): p for p in self.lr_list}
        gt_basenames = {os.path.basename(p): p for p in self.gt_list}
        prompt_embeds_basenames = {os.path.basename(p): p for p in self.prompt_embeds_path_list}
        pooled_prompt_embeds_basenames = {os.path.basename(p): p for p in self.pooled_prompt_embeds_path_list}
        
        # Find common basenames that exist in all required lists (lr, gt, prompt_embeds, pooled_prompt_embeds)
        common_basenames = set(lr_basenames.keys()) & set(gt_basenames.keys()) & set(prompt_embeds_basenames.keys()) & set(pooled_prompt_embeds_basenames.keys())
        
        # Sort to ensure consistent ordering
        common_basenames = sorted(common_basenames)
        
        # Rebuild all lists with only common files
        self.lr_list = [lr_basenames[name] for name in common_basenames]
        self.gt_list = [gt_basenames[name] for name in common_basenames]
        self.prompt_embeds_path_list = [prompt_embeds_basenames[name] for name in common_basenames]
        self.pooled_prompt_embeds_path_list = [pooled_prompt_embeds_basenames[name] for name in common_basenames]

        self.unpair_list = [False] * len(self.lr_list)


        # Verify that we have matching files
        if len(self.lr_list) == 0:
            raise ValueError(
                f"No matching files found across all directories. "
                f"lr_files: {len(lr_basenames)}, gt_files: {len(gt_basenames)}, "
                f"prompt_embeds_files: {len(prompt_embeds_basenames)}, pooled_prompt_embeds_files: {len(pooled_prompt_embeds_basenames)}. "
                f"Common files: {len(common_basenames)}"
            )
        
        # Final consistency check (should always pass after the filtering above)
        if not (len(self.lr_list) == len(self.gt_list) == len(self.prompt_embeds_path_list) == len(self.pooled_prompt_embeds_path_list)):
            raise ValueError(
                f"Internal error: Length mismatch after filtering. "
                f"lr={len(self.lr_list)}, gt={len(self.gt_list)}, "
                f"prompt_embeds={len(self.prompt_embeds_path_list)}, pooled_prompt_embeds={len(self.pooled_prompt_embeds_path_list)}"
            )

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        if lr_only_folder is not None:
            unpair_lr_path = lr_only_folder + '/latent_lr'
            unpair_prompt_embeds_path = lr_only_folder + '/prompt_embeds'
            unpair_pooled_prompt_embeds_path = lr_only_folder + '/pooled_prompt_embeds'

            unpair_lr_list = sorted(glob.glob(os.path.join(unpair_lr_path, '*.pt')))
            unpair_prompt_list = sorted(glob.glob(os.path.join(unpair_prompt_embeds_path, '*.pt')))
            unpair_pooled_prompt_list = sorted(glob.glob(os.path.join(unpair_pooled_prompt_embeds_path, '*.pt')))

            # 去掉 NULL
            if self.null_prompt_embeds_path in unpair_prompt_list:
                unpair_prompt_list.remove(self.null_prompt_embeds_path)
            if self.null_pooled_prompt_embeds_path in unpair_pooled_prompt_list:
                unpair_pooled_prompt_list.remove(self.null_pooled_prompt_embeds_path)

            # basename 对齐（只有 lr / prompt / pooled）
            unpair_lr_basenames = {os.path.basename(p): p for p in unpair_lr_list}
            unpair_prompt_basenames = {os.path.basename(p): p for p in unpair_prompt_list}
            unpair_pooled_basenames = {os.path.basename(p): p for p in unpair_pooled_prompt_list}

            unpair_common = sorted(
                set(unpair_lr_basenames) &
                set(unpair_prompt_basenames) &
                set(unpair_pooled_basenames)
            )

            # 追加到主列表
            self.lr_list += [unpair_lr_basenames[k] for k in unpair_common]
            self.gt_list += [unpair_lr_basenames[k] for k in unpair_common]   # 没有 HR 使用 LR 代替
            self.prompt_embeds_path_list += [unpair_prompt_basenames[k] for k in unpair_common]
            self.pooled_prompt_embeds_path_list += [unpair_pooled_basenames[k] for k in unpair_common]
            self.unpair_list += [True] * len(unpair_common)



    def __getitem__(self, index):
       
        gt_path = self.gt_list[index]
        gt_latent = torch.load(gt_path)
        
        lq_path = self.lr_list[index]
        lq_latent = torch.load(lq_path)

        is_unpair = self.unpair_list[index]          # flag 

        if random.random() < self.null_text_ratio and self.null_prompt_embeds is not None:
            # Use pre-loaded NULL embeddings (clone to avoid sharing tensor references)
            prompt_embeds = self.null_prompt_embeds.clone() if isinstance(self.null_prompt_embeds, torch.Tensor) else self.null_prompt_embeds
            pooled_prompt_embeds = self.null_pooled_prompt_embeds.clone() if isinstance(self.null_pooled_prompt_embeds, torch.Tensor) else self.null_pooled_prompt_embeds
        else:
            prompt_embeds_path = self.prompt_embeds_path_list[index]
            prompt_embeds = torch.load(prompt_embeds_path)

            pooled_prompt_embeds_path = self.pooled_prompt_embeds_path_list[index]
            pooled_prompt_embeds = torch.load(pooled_prompt_embeds_path)

        # Ensure tensors have correct shape - remove batch dimension if present
        # but preserve other dimensions
        if lq_latent.dim() > 0 and lq_latent.shape[0] == 1:
            lq_latent = lq_latent.squeeze(0)
        if gt_latent.dim() > 0 and gt_latent.shape[0] == 1:
            gt_latent = gt_latent.squeeze(0)
        if prompt_embeds.dim() > 0 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds.squeeze(0)
        if pooled_prompt_embeds.dim() > 0 and pooled_prompt_embeds.shape[0] == 1:
            pooled_prompt_embeds = pooled_prompt_embeds.squeeze(0)
        
        example = dict()
        example["conditioning_pixel_values"] = lq_latent
        example["pixel_values"] = gt_latent
        example['prompt_embeds'] = prompt_embeds
        example['pooled_prompt_embeds'] = pooled_prompt_embeds

        example["is_unpair"] = is_unpair               # flag 


        return example

    def __len__(self):
        return len(self.gt_list)