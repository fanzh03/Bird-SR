import argparse
import glob
import os

from PIL import Image
import pandas as pd
import torch
from torchvision import transforms


tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])


def to_lpips_input(tensor):
    return (tensor * 2 - 1).clamp(-1, 1)


def load_image_tensor(path, device):
    image = Image.open(path).convert("RGB")
    return tensor_transforms(image).unsqueeze(0).to(device)


def collect_pairs(lr_path, gt_path):
    if os.path.isfile(lr_path) and os.path.isfile(gt_path):
        name = os.path.splitext(os.path.basename(lr_path))[0]
        return [(name, lr_path, gt_path)], []

    lr_files = sorted(glob.glob(os.path.join(lr_path, "*.*")))
    gt_files = sorted(glob.glob(os.path.join(gt_path, "*.*")))
    gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_files}

    pairs = []
    missing = []
    for lr_file in lr_files:
        name = os.path.splitext(os.path.basename(lr_file))[0]
        gt_file = gt_map.get(name)
        if gt_file is None:
            missing.append(name)
            continue
        pairs.append((name, lr_file, gt_file))
    return pairs, missing


def build_lpips(net, lpips_ckpt, backbone_ckpt, device):
    import lpips

    if lpips_ckpt is None and backbone_ckpt is None:
        loss_fn = lpips.LPIPS(net=net, pretrained=True)
    else:
        loss_fn = lpips.LPIPS(net=net, pretrained=False, pnet_rand=False)
        if backbone_ckpt:
            trunk_state = torch.load(backbone_ckpt, map_location="cpu")
            loss_fn.net.load_state_dict(trunk_state, strict=False)
        if lpips_ckpt:
            lpips_state = torch.load(lpips_ckpt, map_location="cpu")
            loss_fn.load_state_dict(lpips_state, strict=False)

    loss_fn.eval()
    loss_fn.requires_grad_(False)
    return loss_fn.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute LPIPS for paired images.")
    parser.add_argument("--image_path_LR", type=str, required=True)
    parser.add_argument("--image_path_GT", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone. Default: alex.",
    )
    parser.add_argument(
        "--lpips_ckpt",
        type=str,
        default=None,
        help="LPIPS linear weights (e.g. alex.pth). If not set, use lpips pretrained.",
    )
    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default=None,
        help="Backbone weights (e.g. alexnet-owt-7be5be79.pth). Optional.",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    lpips_loss = build_lpips(args.lpips_net, args.lpips_ckpt, args.backbone_ckpt, device)

    pairs, missing = collect_pairs(args.image_path_LR, args.image_path_GT)
    if missing:
        print(f"[WARN] {len(missing)} LR images do not have matching GT: {missing[:10]}")

    records = []
    with torch.no_grad():
        for name, lr_file, gt_file in pairs:
            lr_tensor = to_lpips_input(load_image_tensor(lr_file, device))
            gt_tensor = to_lpips_input(load_image_tensor(gt_file, device))
            lpips_score = lpips_loss(lr_tensor, gt_tensor).mean().item()
            records.append({"image": name, "lpips": lpips_score})
            print(f"[LPIPS] {name}: {lpips_score:.6f}")

    df = pd.DataFrame(records)
    if not df.empty:
        mean_lpips = df["lpips"].mean()
        print(f"[LPIPS MEAN] {mean_lpips:.6f}")
        df = pd.concat(
            [df, pd.DataFrame([{"image": "__mean__", "lpips": mean_lpips}])],
            ignore_index=True,
        )

    csv_path = os.path.join(args.output_dir, "lpips_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"LPIPS results saved to {csv_path}")


if __name__ == "__main__":
    main()
