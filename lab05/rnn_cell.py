import numpy as np
import torch
import torch.nn as nn


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_xh = np.random.uniform(-scale, scale, (hidden_size, input_size))
        self.W_hh = np.random.uniform(-scale, scale, (hidden_size, hidden_size))
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x_t, h_prev):
        if x_t.ndim == 1:
            x_t = x_t.reshape(-1, 1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        linear_term = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
        h_t = np.tanh(linear_term)
        return h_t


def verify_against_pytorch(input_size=4, hidden_size=3):
    cell = SimpleRNNCell(input_size, hidden_size)
    x = np.random.randn(input_size, 1)
    h = np.zeros((hidden_size, 1))
    out_custom = cell.forward(x, h)

    torch_cell = nn.RNNCell(input_size, hidden_size)
    with torch.no_grad():
        torch_cell.weight_ih.copy_(torch.tensor(cell.W_xh).float())
        torch_cell.weight_hh.copy_(torch.tensor(cell.W_hh).float())
        torch_cell.bias_ih.zero_()
        torch_cell.bias_hh.zero_()

    out_torch = (
        torch_cell(torch.tensor(x.T).float(), torch.tensor(h.T).float())
        .detach()
        .numpy()
        .T
    )
    diff = float(np.max(np.abs(out_custom - out_torch)))
    return diff < 1e-6, diff


if __name__ == "__main__":
    success, diff = verify_against_pytorch()
    print(f"Verification status: {success}, max absolute diff: {diff}")
