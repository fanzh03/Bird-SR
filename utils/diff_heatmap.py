import argparse
from pathlib import Path
import cv2
import numpy as np
import random


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk and return it in RGB format."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_difference(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute absolute per-pixel difference magnitude between two RGB images."""
    if gt.shape != pred.shape:
        raise ValueError(
            f"Image shape mismatch: gt.shape={gt.shape}, pred.shape={pred.shape}. "
            "Resize images to the same resolution before generating the heatmap."
        )

    diff = np.abs(gt.astype(np.float32) - pred.astype(np.float32))
    # Collapse channel-wise difference into a single magnitude map (L2 norm)
    diff_magnitude = np.linalg.norm(diff, axis=2)
    # Normalize difference to [0, 1]
    diff_norm = diff_magnitude / (diff_magnitude.max() + 1e-8)
    return diff_norm


def apply_random_perturbations(diff_map: np.ndarray, intensity: float = 0.0) -> np.ndarray:
    """
    Apply random perturbations to the difference map with strong effects on high-value regions.
    
    Args:
        diff_map: Normalized difference map (0-1 range).
        intensity: Perturbation intensity from 0 to 1. Higher values mean stronger perturbations.
    
    Returns:
        Perturbed difference map.
    """
    if intensity <= 0:
        return diff_map
    
    if not (0 <= intensity <= 1):
        raise ValueError(f"Intensity must be in [0, 1], got {intensity}")
    
    perturbed = diff_map.copy()
    h, w = diff_map.shape
    
    # List of available perturbation functions
    perturbations = []
    
    # 2. High-value region reduction - 高值区域减弱（使用平滑过渡而非直接置零）
    def apply_high_value_reduction(img):
        threshold = 0.3 + (1 - intensity) * 0.4
        high_value_mask = img > threshold
        
        if np.any(high_value_mask):
            num_reductions = int(2 + intensity * 5)
            reduced = img.copy()
            
            for _ in range(num_reductions):
                high_y, high_x = np.where(high_value_mask)
                if len(high_y) > 0:
                    idx = random.randint(0, len(high_y) - 1)
                    center_y, center_x = high_y[idx], high_x[idx]
                    
                    radius = int(min(h, w) * (0.08 + intensity * 0.12))
                    y, x = np.ogrid[:h, :w]
                    dist_sq = (x - center_x)**2 + (y - center_y)**2
                    sigma = radius / 2.0
                    weight = np.exp(-dist_sq / (2 * sigma**2))
                    
                    kernel_size = int(5 + intensity * 15)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    blurred = cv2.GaussianBlur(reduced, (kernel_size, kernel_size), intensity * 8.0)
                    
                    reduction_factor = 0.1 + (1 - intensity) * 0.3
                    reduced = reduced * (1 - weight) + (reduced * reduction_factor + blurred * (1 - reduction_factor)) * weight
            
            blend_ratio = 0.4 + intensity * 0.4
            return img * (1 - blend_ratio) + reduced * blend_ratio
        return img
    
    # 3. Local warping in random regions - 在随机局部区域应用扭曲
    def apply_strong_warp(img):
        if h < 10 or w < 10:
            return img
        
        result = img.copy()
        num_regions = random.randint(1, 4)
        
        for _ in range(num_regions):
            center_x = random.randint(int(w * 0.2), int(w * 0.8))
            center_y = random.randint(int(h * 0.2), int(h * 0.8))
            
            region_size = int(min(h, w) * (0.15 + random.random() * 0.25))
            
            local_intensity = intensity * (0.5 + random.random() * 0.5)
            
            y, x = np.ogrid[:h, :w]
            dist_sq = (x - center_x)**2 + (y - center_y)**2
            sigma = region_size / 2.5
            region_mask = np.exp(-dist_sq / (2 * sigma**2))
            
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            max_displacement = local_intensity * min(h, w) * 0.15
            dx = np.random.normal(0, max_displacement, (h, w)).astype(np.float32) * region_mask
            dy = np.random.normal(0, max_displacement, (h, w)).astype(np.float32) * region_mask
            
            blur_size = int(15 + local_intensity * 25)
            if blur_size % 2 == 0:
                blur_size += 1
            dx = cv2.GaussianBlur(dx, (blur_size, blur_size), blur_size / 3.0)
            dy = cv2.GaussianBlur(dy, (blur_size, blur_size), blur_size / 3.0)
            
            map_x = (grid_x + dx).astype(np.float32)
            map_y = (grid_y + dy).astype(np.float32)
            warped = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            result = result * (1 - region_mask) + warped * region_mask
        
        return result
    
    # 5. Threshold-based high-value reduction - 基于阈值的高值减弱（平滑过渡）
    def apply_threshold_reduction(img):
        threshold = 0.5 - intensity * 0.4
        mask = img > threshold
        
        if np.any(mask):
            kernel_size = int(7 + intensity * 20)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), intensity * 12.0)
            
            reduction_ratio = 0.3 + intensity * 0.5
            num_pixels = int(np.sum(mask) * reduction_ratio)
            
            if num_pixels > 0:
                y_coords, x_coords = np.where(mask)
                indices = np.random.choice(len(y_coords), num_pixels, replace=False)
                result = img.copy()
                reduction_factor = 0.15 + (1 - intensity) * 0.25
                result[y_coords[indices], x_coords[indices]] = (
                    img[y_coords[indices], x_coords[indices]] * reduction_factor +
                    blurred[y_coords[indices], x_coords[indices]] * (1 - reduction_factor)
                )
                return result
        
        return img
    
    # 7. Local rotation in random regions - 在随机局部区域应用旋转
    def apply_strong_rotation(img):
        result = img.copy()
        num_regions = random.randint(1, 3)
        
        for _ in range(num_regions):
            center_x = random.randint(int(w * 0.2), int(w * 0.8))
            center_y = random.randint(int(h * 0.2), int(h * 0.8))
            
            region_size = int(min(h, w) * (0.2 + random.random() * 0.3))
            
            local_intensity = intensity * (0.5 + random.random() * 0.5)
            max_angle = local_intensity * 15.0
            
            if max_angle > 0.5:
                angle = random.uniform(-max_angle, max_angle)
                
                y, x = np.ogrid[:h, :w]
                dist_sq = (x - center_x)**2 + (y - center_y)**2
                sigma = region_size / 2.5
                region_mask = np.exp(-dist_sq / (2 * sigma**2))
                
                center = (center_x, center_y)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                
                result = result * (1 - region_mask) + rotated * region_mask
        
        return result
    
    # 8. Local elastic deformation - 在随机局部区域应用弹性变形
    def apply_elastic_deformation(img):
        if h < 20 or w < 20:
            return img
        
        result = img.copy()
        num_regions = random.randint(1, 3)
        
        for _ in range(num_regions):
            center_x = random.randint(int(w * 0.2), int(w * 0.8))
            center_y = random.randint(int(h * 0.2), int(h * 0.8))
            
            region_size = int(min(h, w) * (0.2 + random.random() * 0.3))
            
            local_intensity = intensity * (0.5 + random.random() * 0.5)
            
            y, x = np.ogrid[:h, :w]
            dist_sq = (x - center_x)**2 + (y - center_y)**2
            sigma = region_size / 2.5
            region_mask = np.exp(-dist_sq / (2 * sigma**2))
            
            alpha = local_intensity * min(h, w) * 0.3
            sigma_blur = 10 + local_intensity * 20
            
            dx = np.random.randn(h, w).astype(np.float32) * alpha * region_mask
            dy = np.random.randn(h, w).astype(np.float32) * alpha * region_mask
            
            dx = cv2.GaussianBlur(dx, (0, 0), sigma_blur)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma_blur)
            
            x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x_grid + dx).astype(np.float32)
            map_y = (y_grid + dy).astype(np.float32)
            
            deformed = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            result = result * (1 - region_mask) + deformed * region_mask
        
        return result
    
    # 9. High-value region blur and fade - 高值区域模糊并淡化
    def apply_high_value_fade(img):
        threshold = 0.4
        high_mask = img > threshold
        
        if np.any(high_mask):
            kernel_size = int(5 + intensity * 20)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), intensity * 10.0)
            
            fade_factor = 0.2 + (1 - intensity) * 0.3
            result = img.copy()
            result[high_mask] = blurred[high_mask] * fade_factor
            return result
        return img
    
    perturbations = [
        apply_high_value_reduction,
        apply_strong_warp,
        apply_threshold_reduction,
        apply_strong_rotation,
        apply_elastic_deformation,
        apply_high_value_fade,
    ]
    
    num_perturbations = random.randint(1, min(3, len(perturbations)))
    selected = random.sample(perturbations, num_perturbations)
    
    for pert_func in selected:
        perturbed = pert_func(perturbed)
    
    perturbed = np.clip(perturbed, 0, 1)
    
    return perturbed
def create_heatmap(diff_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Map the normalized difference map (0-1) to a RGB heatmap."""
    diff_uint8 = np.uint8(np.clip(diff_map * 255.0, 0, 255))
    heatmap_bgr = cv2.applyColorMap(diff_uint8, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    background: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay heatmap on background image using alpha blending."""
    if not (0 <= alpha <= 1):
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
    return (background * (1 - alpha) + heatmap * alpha).astype(np.uint8)


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Save an RGB image to disk (converted to BGR for OpenCV)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a heatmap overlay highlighting pixel differences between GT and reconstructed images."
    )
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth image path.")
    parser.add_argument("--pred", type=Path, required=True, help="Reconstructed/predicted image path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diff_heatmap.png"),
        help="Output path for the heatmap overlay image.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha blending weight for the heatmap overlay (0=no heatmap, 1=only heatmap).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="OpenCV colormap name (e.g., jet, hot, magma, plasma).",
    )
    parser.add_argument(
        "--perturb-intensity",
        type=float,
        default=0.0,
        help="Intensity of random perturbations to apply to diff_map (0.0=no perturbation, 1.0=maximum). "
        "Perturbations include smoothing, noise, rotation, warping, scaling, etc.",
    )
    return parser.parse_args()


def colormap_from_name(name: str) -> int:
    name_lower = name.lower()
    available = {
        "autumn": cv2.COLORMAP_AUTUMN,
        "bone": cv2.COLORMAP_BONE,
        "jet": cv2.COLORMAP_JET,
        "winter": cv2.COLORMAP_WINTER,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "ocean": cv2.COLORMAP_OCEAN,
        "summer": cv2.COLORMAP_SUMMER,
        "spring": cv2.COLORMAP_SPRING,
        "cool": cv2.COLORMAP_COOL,
        "hsv": cv2.COLORMAP_HSV,
        "pink": cv2.COLORMAP_PINK,
        "hot": cv2.COLORMAP_HOT,
        "parula": cv2.COLORMAP_PARULA,
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "cividis": cv2.COLORMAP_CIVIDIS,
        "twilight": cv2.COLORMAP_TWILIGHT,
        "twilight_shifted": cv2.COLORMAP_TWILIGHT_SHIFTED,
        "turbo": cv2.COLORMAP_TURBO,
    }
    if name_lower not in available:
        valid = ", ".join(sorted(available.keys()))
        raise ValueError(f"Unsupported colormap '{name}'. Available options: {valid}")
    return available[name_lower]


def main() -> None:
    args = parse_args()

    gt_image = load_image(args.gt)
    pred_image = load_image(args.pred)

    diff_map = compute_difference(gt_image, pred_image)
    
    if args.perturb_intensity > 0:
        diff_map = apply_random_perturbations(diff_map, intensity=args.perturb_intensity)
    
    heatmap = create_heatmap(diff_map, colormap_from_name(args.colormap))
    overlay = overlay_heatmap(gt_image, heatmap, alpha=args.alpha)

    save_image(overlay, args.output)


if __name__ == "__main__":
    main()

# python utils/diff_heatmap.py \
# --gt /mnt/shared-storage-user/sciprismax/liuyidi/Data/noise_SR/gt/0000001.png \
# --pred /mnt/shared-storage-user/sciprismax/liuyidi/DiT4SR-main/result/noise/sample00/0000001.png \
# --output /mnt/shared-storage-user/sciprismax/liuyidi/DiT4SR-main/mask/overlay02.png \
# --alpha 0.7 \
# --colormap turbo \
# --perturb-intensity 0.2