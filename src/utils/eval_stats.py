import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return (pred == target).mean()

def roc_auc(logits, labels, save_dir):
    auroc = roc_auc_score(labels, logits)
    fpr, tpr, thresholds = roc_curve(labels, logits)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0,1], [0,1], "k--", label = "Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_dir)
    return auroc

def f1(pred: np.ndarray, target: np.ndarray) -> float:
    return f1_score(target.flatten(), pred.flatten(), average="macro", zero_division=0)

def aggregate_metrics(all_metrics):
    keys = all_metrics[0].keys()
    results = {}
    for k in keys:
        values = [m[k] for m in all_metrics]
        results[k] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
    return results

def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

def compute_fid(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def compute_mmd(real_features: np.ndarray, gen_features: np.ndarray, gamma: float | None = None) -> float:
    if gamma is None:
        gamma = 1.0 / real_features.shape[1]

    def rbf(x, y):
        dists_sq = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-gamma * dists_sq)

    xx = rbf(real_features, real_features).mean()
    yy = rbf(gen_features, gen_features).mean()
    xy = rbf(real_features, gen_features).mean()
    return float(xx + yy - 2 * xy)

def compute_ssim_1d(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:

    C = pred.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.view(1, 1, -1).expand(C, -1, -1)
    pad = window_size // 2

    mu_x = F.conv1d(pred, kernel, padding=pad, groups=C)
    mu_y = F.conv1d(target, kernel, padding=pad, groups=C)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv1d(pred ** 2, kernel, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv1d(target ** 2, kernel, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv1d(pred * target, kernel, padding=pad, groups=C) - mu_xy

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return ssim_map.mean(dim=2)

def ssim_window_size(sf: int) -> int:
    ws = int(11 * sf / 250)
    return ws | 1

def compute_reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, sf: int = 250) -> dict[str, torch.Tensor]:
    mse = (pred - target).pow(2).mean(dim=2)
    mae = (pred - target).abs().mean(dim=2)
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))
    ssim = compute_ssim_1d(pred, target, window_size=ssim_window_size(sf))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}

def forecast_metrics(pred: np.ndarray, gt: np.ndarray, context: np.ndarray | None = None) -> dict:
    n = min(len(pred), len(gt))
    if n == 0:
        return {k: float("nan") for k in ("mse", "mae", "mase", "pearson_r", "snr_db")}
    p, g = pred[:n].astype(np.float64), gt[:n].astype(np.float64)
    err = p - g
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    naive_mae = float(np.mean(np.abs(np.diff(context.astype(np.float64))))) if context is not None and len(context) > 1 else float("nan")
    mase = mae / naive_mae if naive_mae > 0 else float("nan")
    r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 0 and g.std() > 0 else 0.0
    snr = float(10 * np.log10(np.mean(g**2) / np.mean(err**2))) if np.mean(err**2) > 0 else float("inf")
    return {"mse": mse, "mae": mae, "mase": mase, "pearson_r": r if not np.isnan(r) else 0.0, "snr_db": snr}