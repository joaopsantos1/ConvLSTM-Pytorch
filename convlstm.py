from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        peephole: bool = True,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if any(size % 2 == 0 for size in kernel_size):
            raise ValueError("kernel_size deve conter apenas valores ímpares.")

        padding = tuple(size // 2 for size in kernel_size)
        self.hidden_dim = hidden_dim
        self.peephole = peephole

        self.input_conv = nn.Conv2d(input_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_conv = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.input_dropout = nn.Dropout2d(cnn_dropout)
        self.hidden_dropout = nn.Dropout2d(rnn_dropout)

        self.gate_norm = nn.GroupNorm(1, 4 * hidden_dim) if layer_norm else nn.Identity()
        self.cell_norm = nn.GroupNorm(1, hidden_dim) if layer_norm else nn.Identity()

        if peephole:
            self.weight_ci = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
            self.weight_cf = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
            self.weight_co = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))

    def forward(
        self, input_tensor: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        gates = self.gate_norm(
            self.input_conv(self.input_dropout(input_tensor))
            + self.hidden_conv(self.hidden_dropout(hidden))
        )
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)

        if self.peephole:
            input_gate = input_gate + self.weight_ci * cell
            forget_gate = forget_gate + self.weight_cf * cell

        next_cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        next_cell = self.cell_norm(next_cell)

        if self.peephole:
            output_gate = output_gate + self.weight_co * next_cell

        next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
        return next_hidden, next_cell

    def initial_state(
        self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.hidden_dim, height, width)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)
