"""Configuração central do experimento.

Os arquivos deste projeto já contêm sequências prontas no formato
``(amostras, 11, canais, altura, largura)``. Por isso, o experimento usa os
primeiros 5 frames como entrada e prevê os 6 frames seguintes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


DATA_FILES = {
    "taasrad": {
        "directory": "data/TAASRAD",
        "train": ("taasrad19_2010_2011_2012_2013_2014_2015_2016_120x120.npy",),
        "val": ("taasrad19_2017_120x120.npy",),
        "test": ("taasrad19_2018_2019_120x120.npy",),
    },
    "meteonet": {
        "directory": "data/Meteonet",
        "train": ("meteonet_NW_2016_2017_120x120_clean.npy",),
        "val": ("meteonet_NW_2017_limit_09_120x120_clean.npy",),
        "test": ("meteonet_NW_2018_120x120_clean.npy",),
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Hiperparâmetros e caminhos usados por ``train.py``."""

    source: str

    # Dados: cada amostra .npy tem 11 passos temporais.
    input_len: int = 5
    output_len: int = 6
    normalize: bool = True
    norm_cache_dir: str = "data/norm_stats"

    # Modelo.
    hidden_dims: Tuple[int, ...] = (64, 64)
    kernel_size: Tuple[int, int] = (3, 3)
    peephole: bool = True
    layer_norm: bool = True
    cnn_dropout: float = 0.2
    rnn_dropout: float = 0.2

    # Treino.
    batch_size: int = 5
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    teacher_forcing_ratio: float = 0.5
    seeds: Tuple[int, ...] = (0, )
    num_workers: int = 0
    device: str = "cuda"

    # Saídas.
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
