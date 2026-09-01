"""Modelo encoder-forecaster ConvLSTM para prever imagens de radar."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from convlstm import ConvLSTMCell


def make_cell_stack(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    kernel_size: Tuple[int, int],
    cnn_dropout: float,
    rnn_dropout: float,
    peephole: bool,
    layer_norm: bool,
) -> nn.ModuleList:
    """Cria camadas ConvLSTM, ligando a saída de uma à entrada da próxima."""
    cells = []
    channels = input_dim
    for hidden_dim in hidden_dims:
        cells.append(
            ConvLSTMCell(
                channels, hidden_dim, kernel_size, cnn_dropout, rnn_dropout, peephole, layer_norm
            )
        )
        channels = hidden_dim
    return nn.ModuleList(cells)


class EncoderForecasterConvLSTM(nn.Module):
    """Codifica frames observados e prevê os próximos frames autoregressivamente.

    Entrada: ``(lote, passos_de_entrada, canais, altura, largura)``.
    Saída: ``(lote, passos_de_saída, canais, altura, largura)``.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Tuple[int, ...] = (64, 64),
        kernel_size: Tuple[int, int] = (3, 3),
        output_len: int = 6,
        cnn_dropout: float = 0.2,
        rnn_dropout: float = 0.2,
        peephole: bool = True,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims deve ter ao menos uma camada.")

        self.input_dim = input_dim
        self.output_len = output_len
        cell_args = (input_dim, hidden_dims, kernel_size, cnn_dropout, rnn_dropout, peephole, layer_norm)
        self.encoder_cells = make_cell_stack(*cell_args)
        self.forecaster_cells = make_cell_stack(*cell_args)
        self.readout = nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1)

    @staticmethod
    def _run_stack(
        cells: nn.ModuleList,
        frame: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Propaga um frame por todas as camadas e atualiza seus estados."""
        for index, cell in enumerate(cells):
            states[index] = cell(frame, states[index])
            frame = states[index][0]
        return frame, states

    def _initial_states(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batch_size, _, _, height, width = x.shape
        return [
            cell.initial_state(batch_size, height, width, x.device, x.dtype) for cell in self.encoder_cells
        ]

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != self.input_dim:
            raise ValueError("x deve ter formato (lote, tempo, canais, altura, largura).")
        if target is not None and target.shape[1] < self.output_len:
            raise ValueError("target não possui passos suficientes para teacher forcing.")

        states = self._initial_states(x)
        for frame in x.unbind(dim=1):
            _, states = self._run_stack(self.encoder_cells, frame, states)

        forecasts = []
        frame = x[:, -1]
        for step in range(self.output_len):
            hidden, states = self._run_stack(self.forecaster_cells, frame, states)
            prediction = self.readout(hidden)
            forecasts.append(prediction)

            use_target = (
                target is not None
                and self.training
                and torch.rand((), device=x.device) < teacher_forcing_ratio
            )
            frame = target[:, step] if use_target else prediction

        return torch.stack(forecasts, dim=1)
