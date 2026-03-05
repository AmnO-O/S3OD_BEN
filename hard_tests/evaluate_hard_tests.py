from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Tuple, List, Set, Union, Sequence

import numpy as np
from PIL import Image
import torch
import os, sys, importlib
import cv2
from tqdm.auto import tqdm
import pandas as pd


# Helper functions from notebook cells

# From cell sTVpqkPybvoF
def _strip_prefix_if_present(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd

def _cleanup_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # các prefix hay gặp khi train bằng lightning / ddp / wrapper
    for pref in ("state_dict.", "model.", "net.", "module.", "student.", "teacher."):
        sd = _strip_prefix_if_present(sd, pref)
    return sd

def load_s3od_state_dict(ckpt_path: str | Path) -> Dict[str, torch.Tensor]:
    """
    Load state_dict từ checkpoint (.ckpt/.pt) hoặc raw state_dict.
    Hỗ trợ các format phổ biến:
      - {"state_dict": {...}}  (Lightning)
      - {"model_state_dict": {...}}
      - raw: { "layer.weight": Tensor, ... }
      - nested: {"model": {...}} / {"net": {...}} / {"weights": {...}}
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj: Any = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if not isinstance(obj, dict):
        raise ValueError(f"Unrecognized checkpoint type: {type(obj)}")

    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        sd = obj["model_state_dict"]
    elif all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = obj  # raw state_dict
    else:
        # tìm nested dict có tensor
        for k in ("model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict) and all(isinstance(v, torch.Tensor) for v in obj[k].values()):
                sd = obj[k]
                break
        else:
            raise ValueError(f"Unrecognized checkpoint format. Keys: {list(obj.keys())[:30]}")

    return _cleanup_state_dict_keys(sd)

# From cell mD-9B-D6cIqw
def load_weights_into_background_removal(
    detector,
    ckpt_path: str | Path,
    device: str = "cuda",
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Nạp checkpoint vào detector.model (S3OD).
    Trả về (missing_keys, unexpected_keys).
    """
    sd = load_s3od_state_dict(ckpt_path)

    missing, unexpected = detector.model.load_state_dict(sd, strict=strict)
    detector.model.to(device)
    detector.model.eval()

    return missing, unexpected

# From cell 1R2iFTNTcMTu
def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_mask_rgba(output_dir: Path, rel_noext: Path, mask01: np.ndarray, rgba: Image.Image):
    mask_path = output_dir / "masks" / (str(rel_noext) + ".png")
    rgba_path = output_dir / "rgba"  / (str(rel_noext) + ".png")
    ensure_parent(mask_path)
    ensure_parent(rgba_path)

    mask_u8 = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(mask_u8).convert("L").save(mask_path)
    rgba.save(rgba_path)

def run_infer_s3od_to_outputs(
    detector,
    images_dir: str | Path,
    output_dir: str | Path = "outputs",
    threshold: float = 0.5,
    use_refiner: bool = False,
    recursive: bool = True,
):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(images_dir, recursive=recursive)
    if not imgs:
        print(f"No images found in {images_dir}")
        return

    for p in tqdm(imgs, desc="Running Inference"):
        rel_noext = p.relative_to(images_dir).with_suffix("")
        img = Image.open(p).convert("RGB")

        res = detector.remove_background(img, threshold=threshold, use_refiner=use_refiner)

        # res.predicted_mask: float [0..1], res.rgba_image: PIL RGBA
        save_mask_rgba(output_dir, rel_noext, res.predicted_mask, res.rgba_image)

    print("Done ->", output_dir)

# From cell kfAyLrZ1q7De
def save_and_rank_metrics(metrics: dict, run_name: str, filename: str = "s3od_metrics_results.csv") -> pd.DataFrame:
    """
    Saves the current run's metrics to a CSV file and displays ranked results.

    Args:
        metrics: Dictionary of metrics (e.g., from compute_metrics_from_saved_preds).
        run_name: A string identifier for the current run (e.g., 'DIS-TE1_ModelA').
        filename: Name of the CSV file to store results.

    Returns:
        A DataFrame containing all stored results, sorted by relevant metrics.
    """
    metrics_data = {"Run Name": [run_name]}
    for k, v in metrics.items():
        if not k.startswith("_") and isinstance(v, (float, np.ndarray, np.float32, np.float64)):
            metrics_data[k] = [float(v)]

    current_run_df = pd.DataFrame(metrics_data)

    if os.path.exists(filename):
        all_results_df = pd.read_csv(filename)
        all_results_df = pd.concat([all_results_df, current_run_df], ignore_index=True)
    else:
        all_results_df = current_run_df

    # Remove duplicate run_names, keeping the latest one
    all_results_df.drop_duplicates(subset=["Run Name"], keep="last", inplace=True)
    all_results_df.reset_index(drop=True, inplace=True)

    # Save the updated results
    all_results_df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

    # Display ranked results
    print("\n--- Ranked Results ---")
    print("Sorted by MAE (lower is better):")
    print(all_results_df.sort_values(by="MAE", ascending=True).set_index("Run Name"))
    print("\nSorted by MaxF (higher is better):")
    print(all_results_df.sort_values(by="MaxF", ascending=False).set_index("Run Name"))
    print("\nSorted by AvgF (higher is better):")
    print(all_results_df.sort_values(by="AvgF", ascending=False).set_index("Run Name"))

    return all_results_df


# Helper functions from cell RyvGAl-0FqT_
DEFAULT_IMG_EXTS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

def list_images(
    input_dir: Union[str, Path],
    exts: Optional[Iterable[str]] = None,
    recursive: bool = True,
    return_relative: bool = False,
) -> List[Path]:
    """
    List image files inside input_dir.

    Args:
        input_dir: folder chứa ảnh
        exts: iterable các extension (ví dụ [".jpg",".png"]). None => dùng DEFAULT_IMG_EXTS
        recursive: True => quét đệ quy (rglob), False => chỉ quét 1 tầng (glob)
        return_relative: True => trả Path tương đối so với input_dir

    Returns:
        Danh sách Path (đã sort) của các file ảnh.
    """
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"input_dir not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {root}")

    extset = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (exts or DEFAULT_IMG_EXTS)}

    it = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in it if p.is_file() and p.suffix.lower() in extset]
    files.sort()

    if return_relative:
        return [p.relative_to(root) for p in files]
    return files

def find_gt(
    rel_noext: Union[str, Path],
    gt_dir: Union[str, Path],
    gt_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
    name_variants: Sequence[str] = ("", "_mask", "-mask", "_gt", "-gt", "_alpha", "-alpha"),
) -> Optional[Path]:
    """
    Find GT file under gt_dir matching the relative-no-extension path.

    Example:
      rel_noext = Path("scene1/img_0001")   # no suffix
      try gt_dir/scene1/img_0001.png, img_0001_mask.png, img_0001.jpg, ...

    Args:
        rel_noext: Path relative to images_dir without suffix (like your rel_noext)
        gt_dir: root folder containing GT masks
        gt_exts: GT possible extensions to try
        name_variants: suffix variants to try on stem

    Returns:
        Path to GT if found else None
    """
    rel_noext = Path(rel_noext)
    gt_dir = Path(gt_dir)

    # Keep folder structure
    search_root = gt_dir / rel_noext.parent
    base = rel_noext.name  # already without extension

    # normalize extensions
    norm_exts = []
    for e in gt_exts:
        e = e.lower()
        if not e.startswith("."):
            e = "." + e
        norm_exts.append(e)

    # Try base + variant + ext
    for var in name_variants:
        for ext in norm_exts:
            cand = search_root / f"{base}{var}{ext}"
            if cand.exists():
                return cand

    # Extra fallback: if GT filenames already include an extension in rel_noext (rare)
    # or if GT uses same base name somewhere else in the same folder
    if search_root.exists():
        for ext in norm_exts:
            cand = search_root / f"{base}{ext}"
            if cand.exists():
                return cand

    return None

def compute_metrics_from_saved_preds(
    repo_root: str | Path,
    images_dir: str | Path,
    gt_dir: str | Path,
    pred_dir: str | Path,
    device: str = "cuda",
    recursive: bool = True,
    run_name: Optional[str] = None,
) -> Dict:
    """
    Read pred_dir/<rel>.png + gt_dir/<rel>.* and compute repo metrics EvaluationMetrics.
    """
    repo_root = Path(repo_root)
    images_dir = Path(images_dir)
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    sys.path.insert(0, str(repo_root / "synth_sod"))
    from model_training.metrics import EvaluationMetrics  # type: ignore

    metric_counter = EvaluationMetrics(device=device)

    imgs = list_images(images_dir, recursive=recursive)
    if len(imgs) == 0:
        raise RuntimeError(f"No images in {images_dir}")

    hit_pred = hit_gt = used = 0

    for p in tqdm(imgs, desc="PASS 2: Metrics"):
        rel_noext = p.relative_to(images_dir).with_suffix("")
        pred_path = pred_dir / rel_noext.with_suffix(".png")
        gt_path = find_gt(rel_noext, gt_dir)

        if pred_path.exists(): hit_pred += 1
        if gt_path is not None: hit_gt += 1

        if (not pred_path.exists()) or (gt_path is None):
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue

        pred01 = pred.astype(np.float32) / 255.0
        gt01 = (gt > 128).astype(np.float32)

        if gt01.shape != pred01.shape:
            gt01 = cv2.resize(gt01, (pred01.shape[1], pred01.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_t = torch.from_numpy(pred01).float().to(device)
        gt_t = torch.from_numpy(gt01).float().to(device)

        metric_counter.step(pred_t, gt_t.clone())
        used += 1

    metrics = metric_counter.compute_metrics()
    metrics["_check"] = {
        "num_images_found": len(imgs),
        "pred_exists": hit_pred,
        "gt_exists": hit_gt,
        "used_in_eval": used,
        "images_dir": str(images_dir),
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "device": device,
    }

    if run_name is None:
        run_name = pred_dir.name
    save_and_rank_metrics(metrics, run_name=run_name)

    return metrics


# Patched preprocess function from cell -RO-H544cNhF
def patched_preprocess_fixed(self, image):
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise TypeError("Input image must be a PIL Image or numpy array")

    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]

    scale = self.image_size / max(h, w)
    nh_target = max(1, int(round(h * scale)))
    nw_target = max(1, int(round(w * scale)))

    resized = cv2.resize(img, (nw_target, nh_target), interpolation=cv2.INTER_LINEAR)

    rh, rw = resized.shape[:2]

    padded = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    top = (self.image_size - rh) // 2
    left = (self.image_size - rw) // 2

    padded[top:top + rh, left:left + rw, :] = resized

    input_tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    pad_info = {
        "height_pad": top,
        "width_pad": left,
        "original_size": (h, w),
        "resized_size": (rh, rw),
        "scale": scale,
    }
    return input_tensor, pad_info


if __name__ == '__main__':
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Change current directory to S3OD root for imports to work correctly
    # (assuming the script is run from S3OD root or path is adjusted)
    # if os.getcwd().split('/')[-1] != 'S3OD':
    #     os.chdir('S3OD') # This would be problematic in a Colab environment if run outside the repo

    # repo_root = Path(".")  # assuming current directory is S3OD
    # ckpt_path = "last_focal_iou_ssim.ckpt"

    # images_dir = "datasets/DIS5K/DIS5K/DIS-TE1/im"
    # gt_dir     = "datasets/DIS5K/DIS5K/DIS-TE1/gt"
    # output_infer_dir = "outputs/DIS-TE1" # Output directory for inference results
    # pred_dir   = output_infer_dir + "/masks" # Predictions for metrics are in the masks subfolder

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Create detector and patch preprocess function
    from s3od import BackgroundRemoval
    detector = BackgroundRemoval(model_id="okupyn/s3od", image_size=1024, device=device)
    detector._preprocess = patched_preprocess_fixed.__get__(detector, type(detector))
    print("Patched detector._preprocess OK.")

    # 2) Load local checkpoint into detector.model
    # missing, unexpected = load_weights_into_background_removal(
    #     detector,
    #     ckpt_path=ckpt_path,
    #     device=device,
    #     strict=False,
    # )
    # print("missing weights:", len(missing), "unexpected weights:", len(unexpected))

    # 3) Run inference and save outputs
    # print(f"Running inference on images in {images_dir}")
    # run_infer_s3od_to_outputs(
    #     detector,
    #     images_dir=images_dir,
    #     output_dir=output_infer_dir,
    #     threshold=0.5,
    #     use_refiner=True,
    # )

    # 4) Compute and rank metrics
    # print("\nComputing metrics...")
    # metrics = compute_metrics_from_saved_preds(
    #     repo_root=repo_root,
    #     images_dir=images_dir,
    #     gt_dir=gt_dir,
    #     pred_dir=pred_dir,
    #     device=device,
    #     run_name="DIS-TE1_S3OD_FineTuned"
    # )
    # print("\nFinal Metrics:")
    # print(metrics)
