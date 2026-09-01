"""Funções pequenas compartilhadas pelo treinamento e pela avaliação."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def set_seed(seed: int) -> None:
    """Inicializa fontes de aleatoriedade para tornar uma execução repetível."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_csv(path: str | Path, rows: list[dict]) -> None:
    """Salva linhas de métricas em CSV, criando o diretório pai quando necessário."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def compute_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Calcula MSE, MAE, RMSE, SSIM e PSNR para um lote de previsões.
    """
    error = prediction - target
    mse = error.square().mean().item()
    mae = error.abs().mean().item()
 
    batch, steps, channels, height, width = prediction.shape
    prediction_frames = prediction.reshape(batch * steps, channels, height, width)
    target_frames = target.reshape(batch * steps, channels, height, width)
    data_range = (target_frames.max() - target_frames.min()).clamp_min(1e-6).item()
    ssim = structural_similarity_index_measure(prediction_frames, target_frames, data_range=data_range).item()
 
    psnr = 10 * torch.log10(target_frames.max().sub(target_frames.min()).clamp_min(1e-6).square() / mse).item()
    return {"mse": mse, "mae": mae, "rmse": mse**0.5, "ssim": ssim, "psnr": psnr}


def summarize_runs(runs: list[dict[str, float]]) -> list[dict[str, float | str]]:
    """Resume métricas numéricas de várias seeds em linhas prontas para CSV."""
    summary = []
    for metric in runs[0]:
        try:
            values = np.asarray([run[metric] for run in runs], dtype=float)
        except (TypeError, ValueError):
            continue
        summary.append({"metric": metric, "mean": float(np.nanmean(values)), "std": float(np.nanstd(values))})
    return summary
