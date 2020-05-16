"""lstm.py
Author: Urchade Zaratiana
"""
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, reverse=False):
        super().__init__()

        self.batch_first = batch_first
        self.reversed = reverse

        self.Wi = nn.Parameter(torch.randn(size=(input_size, hidden_size)))
        self.Wo = nn.Parameter(torch.randn(size=(input_size, hidden_size)))
        self.Wz = torch.nn.Parameter(torch.randn(size=(input_size, hidden_size)))

        self.Ri = nn.Parameter(torch.randn(size=(hidden_size, hidden_size)))
        self.Ro = nn.Parameter(torch.randn(size=(hidden_size, hidden_size)))
        self.Rz = nn.Parameter(torch.randn(size=(hidden_size, hidden_size)))

        self.bi = nn.Parameter(torch.randn(size=(hidden_size,)))
        self.bo = nn.Parameter(torch.randn(size=(hidden_size,)))
        self.bz = nn.Parameter(torch.randn(size=(hidden_size,)))

        self.h_init = torch.zeros(size=(hidden_size,), dtype=torch.float32)
        self.c_init = torch.zeros(size=(hidden_size,), dtype=torch.float32)

    def forward(self, x, states=None):

        if states is None:
            h_t = self.h_init
            c_t = self.c_init
        else:
            h_t, c_t = states

        if self.batch_first:
            _, seq_length, n_input = x.shape
        else:
            seq_length, _, _ = x.shape

        timestep = list(range(seq_length))

        if self.reversed:
            timestep = list(reversed(timestep))

        outputs = []

        for t in timestep:

            if self.batch_first:
                x_t = x[:, t, :].clone()
            else:
                x_t = x[t, :, :].clone()

            input_gate = torch.sigmoid(x_t @ self.Wi + h_t @ self.Ri + self.bi)
            output_gate = torch.sigmoid(x_t @ self.Wo + h_t @ self.Ro + self.bo)
            cell_input = torch.tanh(x_t @ self.Wz + h_t @ self.Rz + self.bz)
            c_t = c_t + input_gate * cell_input
            h_t = output_gate * torch.tanh(c_t)

            outputs.append(h_t)

        outputs = torch.stack(outputs)

        return outputs, (h_t, c_t)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()

        forward = LSTM(input_size, hidden_size, batch_first)
        backward = LSTM(input_size, hidden_size, batch_first, reverse=True)

        self.direction = nn.ModuleList([forward, backward])

        self.hid_init = torch.zeros(size=(hidden_size,), dtype=torch.float32)

    def forward(self, x, states=None):

        if states is None:
            state_forward = self.hid_init, self.hid_init
            state_backward = self.hid_init, self.hid_init
            states = state_forward, state_backward

        else:
            h, c = states
            state_forward = h[0], c[0]
            state_backward = h[1], c[1]
            states = state_forward, state_backward

        outputs = []

        hidden_states = []
        cell_states = []

        for i, layer in enumerate(self.direction):
            state = states[i]
            out, (h, c) = layer(x, state)

            hidden_states.append(h)
            cell_states.append(c)

            outputs.append(out)

        hidden_states = torch.stack(hidden_states)
        cell_states = torch.stack(cell_states)

        outputs = torch.cat(outputs, -1)

        return outputs, (hidden_states, cell_states)
