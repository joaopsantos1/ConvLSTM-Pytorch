"""Treina e avalia o modelo ConvLSTM para uma das fontes de radar.

Exemplos:
    python train.py --source taasrad
    python train.py --source meteonet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import DATA_FILES, ExperimentConfig
from datasets import build_datasets
from model import EncoderForecasterConvLSTM
from utils import compute_metrics, save_csv, set_seed, summarize_runs


def select_device(preferred: str) -> torch.device:
    """Escolhe CUDA quando disponível; caso contrário, segue com CPU."""
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        print("[aviso] CUDA não disponível; usando CPU.")
        return torch.device("cpu")
    return torch.device(preferred)


def make_loader(dataset, batch_size: int, shuffle: bool, config: ExperimentConfig, device: torch.device) -> DataLoader:
    """Centraliza opções iguais dos três DataLoaders."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    teacher_forcing_ratio: float = 0.0,
    progress_label: str | None = None,
) -> float:
    """Executa uma época de treino (com otimizador) ou validação (sem ele)."""
    training = optimizer is not None
    model.train(training)
    loss_function = torch.nn.MSELoss()
    total_loss = 0.0
    sample_count = 0
    batches = loader
    if progress_label:
        batches = tqdm(loader, desc=progress_label, unit="lote", dynamic_ncols=True)

    with torch.set_grad_enabled(training):
        for inputs, targets in batches:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = model(
                inputs,
                target=targets if training else None,
                teacher_forcing_ratio=teacher_forcing_ratio if training else 0.0,
            )
            loss = loss_function(prediction, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)

            if progress_label:
                batches.set_postfix(mse=f"{total_loss / sample_count:.6f}")

    return total_loss / sample_count


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalization: dict[str, float],
) -> dict[str, float]:
    """Calcula métricas de teste ponderadas pela quantidade de amostras.

    Previsões e alvos são desnormalizados (``x * std + mean``) antes do
    cálculo, usando as estatísticas de normalização do treino, para que
    mse/mae/rmse/ssim/psnr sejam reportados na unidade física original do
    radar em vez da escala padronizada usada internamente pelo modelo.
    """
    model.eval()
    mean = normalization["mean"]
    std = normalization["std"]
    totals: dict[str, float] = {}
    inference_seconds = 0.0
    sample_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        start = time.perf_counter()
        prediction = model(inputs)
        inference_seconds += time.perf_counter() - start

        prediction_denorm = prediction * std + mean
        targets_denorm = targets * std + mean

        batch_size = inputs.size(0)
        sample_count += batch_size
        for name, value in compute_metrics(prediction_denorm, targets_denorm).items():
            totals[name] = totals.get(name, 0.0) + value * batch_size

    metrics = {name: value / sample_count for name, value in totals.items()}
    metrics["total_inference_time_s"] = inference_seconds
    metrics["avg_infer_time_per_sample_s"] = inference_seconds / sample_count
    return metrics


def emissions_metrics(tracker: EmissionsTracker, emissions_kg: float | None) -> dict[str, float | None]:
    """Extrai as métricas energéticas do CodeCarbon, quando disponibilizadas pela plataforma."""
    data = tracker.final_emissions_data

    def value(name: str) -> float | None:
        item = getattr(data, name, None) if data is not None else None
        if item is None:
            return None
        try:
            return float(item)
        except (TypeError, ValueError):
            return None

    return {
        "co2_emissions_kg": emissions_kg,
        "energy_consumed_kwh": value("energy_consumed"),
        "gpu_energy_kwh": value("gpu_energy"),
        "cpu_energy_kwh": value("cpu_energy"),
        "ram_energy_kwh": value("ram_energy"),
        "codecarbon_duration_s": value("duration"),
    }


def train_one_seed(
    config: ExperimentConfig,
    seed: int,
    datasets: tuple,
    normalization: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    """Treina uma seed, recupera o melhor checkpoint e o avalia no teste."""
    set_seed(seed)
    train_data, validation_data, test_data = datasets
    train_loader = make_loader(train_data, config.batch_size, True, config, device)
    validation_loader = make_loader(validation_data, config.batch_size, False, config, device)
    test_loader = make_loader(test_data, config.batch_size, False, config, device)

    model = EncoderForecasterConvLSTM(
        hidden_dims=config.hidden_dims,
        kernel_size=config.kernel_size,
        output_len=config.output_len,
        cnn_dropout=config.cnn_dropout,
        rnn_dropout=config.rnn_dropout,
        peephole=config.peephole,
        layer_norm=config.layer_norm,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    checkpoint_path = Path(config.checkpoint_dir) / config.source / f"seed_{seed}" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    total_train_seconds = 0.0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tracker = EmissionsTracker(
        project_name=f"convlstm-{config.source}-seed-{seed}",
        save_to_file=False,
        log_level="error",
    )
    tracker.start()

    for epoch in range(config.epochs):
        teacher_forcing = config.teacher_forcing_ratio * max(0.0, 1 - epoch / max(config.epochs - 1, 1))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        progress_prefix = f"{config.source} | seed {seed} | época {epoch + 1}/{config.epochs}"
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            config.grad_clip,
            teacher_forcing,
            progress_label=f"{progress_prefix} | treino",
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            device,
            progress_label=f"{progress_prefix} | validação",
        )
        elapsed = time.perf_counter() - start
        total_train_seconds += elapsed
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        completed_epochs = epoch + 1
        average_epoch_seconds = total_train_seconds / completed_epochs
        eta_seconds = average_epoch_seconds * (config.epochs - completed_epochs)
        memory_info = ""
        if device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            memory_info = f" | VRAM pico={peak_memory_gb:.2f} GB"

        status = "melhor validação" if validation_loss < best_validation_loss else "sem melhora"
        print(
            f"[{config.source} | seed {seed}] época {completed_epochs:03d}/{config.epochs}"
            f" | MSE treino={train_loss:.6f} | MSE validação={validation_loss:.6f}"
            f" | tempo={elapsed:.1f}s | ETA={eta_seconds / 60:.1f} min"
            f" | {status}{memory_info}",
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": validation_loss}, checkpoint_path)

    emissions_kg = tracker.stop()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, test_loader, device, normalization)

    # As perdas de treino/validação acima (train_losses, validation_losses,
    # best_validation_loss) foram calculadas com nn.MSELoss() sobre dados
    # NORMALIZADOS, e continuam assim para a seleção de checkpoint e o
    # agendamento de teacher forcing — nada disso muda.
    #
    # Para os valores logados abaixo, convertemos esse MSE normalizado para
    # a escala física (unidade original do radar). Como pred_norm-target_norm
    # = (pred-target)/std, o MSE em unidades originais é MSE_norm * std²
    # (não um simples "* std + mean", que só vale para valores brutos, não
    # para erros quadráticos).
    variance = normalization["std"] ** 2
    metrics.update(
        seed=seed,
        best_val_loss=best_validation_loss * variance,
        best_epoch=checkpoint["epoch"] + 1,
        total_train_time_s=total_train_seconds,
        trained_epochs=len(train_losses),
        mean_train_mse=float(sum(train_losses) / len(train_losses)) * variance,
        final_train_mse=train_losses[-1] * variance,
        mean_validation_mse=float(sum(validation_losses) / len(validation_losses)) * variance,
        final_validation_mse=validation_losses[-1] * variance,
        avg_epoch_time_s=total_train_seconds / len(train_losses),
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
        peak_gpu_reserved_memory_mib=torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else 0.0,
    )
    metrics.update(emissions_metrics(tracker, emissions_kg))
    metrics_dir = Path(config.results_dir) / config.source
    save_csv(metrics_dir / f"test_metrics_seed_{seed}.csv", [metrics])
    save_csv(metrics_dir / f"execution_metrics_seed_{seed}.csv", [{key: value for key, value in metrics.items() if key not in {"mse", "mae", "rmse", "ssim", "psnr"}}])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina ConvLSTM para nowcasting de radar.")
    parser.add_argument("--source", choices=DATA_FILES, required=True, help="Conjunto de dados a usar.")
    args = parser.parse_args()

    config = ExperimentConfig(source=args.source)
    device = select_device(config.device)
    print(f"Dispositivo: {device}")
    datasets = build_datasets(
        config.source, config.input_len, config.output_len, config.normalize, config.norm_cache_dir
    )
    train_data, validation_data, test_data, normalization = datasets
    print(f"Dados: treino={len(train_data)}, validação={len(validation_data)}, teste={len(test_data)}")
    print(f"Normalização: média={normalization['mean']:.4f}, desvio={normalization['std']:.4f}")

    results = [train_one_seed(config, seed, datasets[:3], normalization, device) for seed in config.seeds]
    summary = summarize_runs(results)
    save_csv(Path(config.results_dir) / config.source / "summary_metrics.csv", summary)

    print(f"\n=== Resumo final: {config.source} ({len(config.seeds)} seeds) ===")
    for row in summary:
        print(f"{row['metric']}: {row['mean']:.4f} ± {row['std']:.4f}")


if __name__ == "__main__":
    main()