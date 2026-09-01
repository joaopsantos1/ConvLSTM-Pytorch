"""Leitura dos arquivos de radar e normalização do conjunto de treino."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATA_FILES


Array = np.memmap
EXPECTED_DIMENSIONS = 5  # (amostras, tempo, canais, altura, largura)


def load_arrays(source: str, split: str) -> list[Array]:
    """Abre os arquivos de um split em modo memória mapeada.

    O modo mmap permite acessar somente o lote necessário, sem carregar todos
    os dados de radar na RAM.
    """
    source_config = DATA_FILES[source]
    paths = [Path(source_config["directory"]) / name for name in source_config[split]]
    arrays = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"{path} tem formato {array.shape}; esperado "
                "(amostras, tempo, canais, altura, largura)."
            )
        arrays.append(array)
    return arrays


class RadarSequenceDataset(Dataset):
    """Separa cada sequência pronta em frames observados e frames-alvo."""

    def __init__(self, arrays: Iterable[Array], input_len: int, output_len: int, mean: float, std: float):
        self.arrays = list(arrays)
        self.input_len = input_len
        self.output_len = output_len
        total_len = input_len + output_len

        self.index = []
        for array_index, array in enumerate(self.arrays):
            if array.shape[1] < total_len:
                raise ValueError(
                    f"Cada amostra tem {array.shape[1]} passos, mas a configuração pede {total_len}."
                )
            self.index.extend((array_index, sample_index) for sample_index in range(len(array)))

        if not self.index:
            raise ValueError("O split não contém amostras.")
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        array_index, sample_index = self.index[index]
        sequence = np.asarray(
            self.arrays[array_index][sample_index, : self.input_len + self.output_len], dtype=np.float32
        )
        sequence = (sequence - self.mean) / self.std
        return torch.from_numpy(sequence[: self.input_len]), torch.from_numpy(sequence[self.input_len :])


def compute_norm_stats(source: str, cache_dir: str, sample_stride: int = 50) -> dict[str, float]:
    """Calcula média/desvio do treino, amostrando sequências para poupar RAM."""
    cache_path = Path(cache_dir) / f"{source}_norm_stats.json"
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8") as file:
            cached = json.load(file)
        if cached.get("format") == "prewindowed-v1":
            return {"mean": cached["mean"], "std": cached["std"]}

    total, total_squared, count = 0.0, 0.0, 0
    for array in load_arrays(source, "train"):
        values = np.asarray(array[::sample_stride], dtype=np.float64)
        total += values.sum()
        total_squared += np.square(values).sum()
        count += values.size

    mean = total / count
    variance = max(total_squared / count - mean**2, 0.0)
    stats = {"format": "prewindowed-v1", "mean": float(mean), "std": float(np.sqrt(variance) + 1e-6)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    return {"mean": stats["mean"], "std": stats["std"]}


def build_datasets(
    source: str, input_len: int, output_len: int, normalize: bool, norm_cache_dir: str
) -> tuple[RadarSequenceDataset, RadarSequenceDataset, RadarSequenceDataset, dict[str, float]]:
    """Constrói os três splits usando as mesmas estatísticas do treino."""
    if source not in DATA_FILES:
        raise ValueError(f"Fonte desconhecida: {source}. Opções: {', '.join(DATA_FILES)}")

    stats = compute_norm_stats(source, norm_cache_dir) if normalize else {"mean": 0.0, "std": 1.0}
    datasets = [
        RadarSequenceDataset(load_arrays(source, split), input_len, output_len, **stats)
        for split in ("train", "val", "test")
    ]
    return *datasets, stats
