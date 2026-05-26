from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer for ConvLSTMCell")
        padding = kernel_size // 2
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        i, f, o, g = torch.chunk(self.gates(combined), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMBaseline(nn.Module):
    """
    ConvLSTM baseline that decodes a multi-step rollout map [B,T,H,W] from static input channels [B,C,H,W].

    The network keeps a static spatial context from inputs and iteratively predicts one frame per step.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 50,
        hidden_channels: int = 48,
        num_layers: int = 2,
        kernel_size: int = 3,
        context_channels: int | None = None,
        use_feedback: bool = True,
    ):
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be > 0 for ConvLSTM rollout decoding")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.use_feedback = bool(use_feedback)
        self.context_channels = int(context_channels or hidden_channels)

        self.context_net = nn.Sequential(
            nn.Conv2d(in_channels, self.context_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.context_channels, self.context_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.init_h = nn.Conv2d(self.context_channels, self.hidden_channels, kernel_size=1)
        self.init_c = nn.Conv2d(self.context_channels, self.hidden_channels, kernel_size=1)

        cells: list[nn.Module] = []
        for i in range(self.num_layers):
            cell_in = self.context_channels if i == 0 else self.hidden_channels
            cells.append(ConvLSTMCell(cell_in, self.hidden_channels, kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

        self.readout = nn.Conv2d(self.hidden_channels, 1, kernel_size=1)
        self.feedback_proj = nn.Conv2d(1, self.context_channels, kernel_size=1) if self.use_feedback else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        context = self.context_net(x)

        h_states = [self.init_h(context) for _ in range(self.num_layers)]
        c_states = [self.init_c(context) for _ in range(self.num_layers)]

        feedback = torch.zeros(
            b,
            self.context_channels,
            h,
            w,
            device=x.device,
            dtype=x.dtype,
        )

        outputs: list[torch.Tensor] = []
        for _ in range(self.out_channels):
            layer_in = context + feedback if self.use_feedback else context
            for i, cell in enumerate(self.cells):
                h_i, c_i = cell(layer_in, h_states[i], c_states[i])
                h_states[i], c_states[i] = h_i, c_i
                layer_in = h_i

            frame = self.readout(layer_in)  # [B,1,H,W]
            outputs.append(frame)
            if self.use_feedback and self.feedback_proj is not None:
                feedback = self.feedback_proj(frame)

        return torch.cat(outputs, dim=1)  # [B,T,H,W]
