# plastic_train/plastic.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- simple utility to keep eta positive and bounded ----
def _eta_from_param(param, learnable: bool, init_eta: float, device):
    if learnable:
        eta = F.softplus(param)  # positive
        return torch.clamp(eta, 1e-5, 0.25)
    else:
        return torch.as_tensor(init_eta, dtype=torch.float32, device=device)

class PlasticLinear(nn.Module):
    """Time-distributed Linear with Hebbian/Oja fast weights."""
    def __init__(self, in_dim, out_dim, init_eta=0.1, learn_eta=False, rule="oja", bias=True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        self.register_buffer("H", torch.zeros(1, out_dim, in_dim))  # fast weights per-batch
        self.update_traces = False
        self.modulators = None

        self.rule = str(rule).lower()
        self.learn_eta = bool(learn_eta)
        if learn_eta:
            self._eta_param = nn.Parameter(torch.log(torch.expm1(torch.tensor(init_eta))))
        else:
            self.register_buffer("_eta_const", torch.tensor(float(init_eta), dtype=torch.float32))

        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def reset_traces(self, batch_size: int, device=None):
        device = device or self.W.device
        self.H = torch.zeros(batch_size, self.out_dim, self.in_dim, device=device)

    def set_mode(self, *, update_traces: bool, modulators: torch.Tensor | None):
        self.update_traces = bool(update_traces)
        self.modulators = modulators

    def forward(self, x):  # x: (B,T,in)
        B, T, Din = x.shape
        assert Din == self.in_dim
        if self.H.dim() != 3 or self.H.size(0) != B:
            self.H = torch.zeros(B, self.out_dim, self.in_dim, device=x.device, dtype=x.dtype)

        eta = _eta_from_param(getattr(self, "_eta_param", None),
                              self.learn_eta, float(getattr(self, "_eta_const", 0.1)), x.device)

        outs = []
        for t in range(T):
            x_t = x[:, t, :]                         # (B,in)
            y_static = F.linear(x_t, self.W, self.bias)  # (B,out)
            y_fast = torch.bmm(self.H, x_t.unsqueeze(-1)).squeeze(-1)
            y_t = y_static + y_fast
            outs.append(y_t)

            if self.update_traces:
                if self.modulators is None:
                    m_t = torch.ones(B, device=x.device, dtype=x.dtype)
                else:
                    m = self.modulators
                    if m.dim() == 1:
                        m_t = m.expand(B)
                    elif m.dim() == 2:
                        m_t = m[:, t]
                    else:
                        raise ValueError("modulators must be (T,), (1,T), or (B,T)")
                    m_t = m_t.to(dtype=x.dtype, device=x.device)

                outer = torch.einsum("bo,bi->boi", y_t, x_t)
                if self.rule == "hebb":
                    dH = eta * m_t.view(B, 1, 1) * outer
                else:  # oja
                    dH = eta * m_t.view(B, 1, 1) * (outer - (y_t.pow(2).unsqueeze(-1) * self.H))
                self.H = self.H + dH

        Y = torch.stack(outs, dim=1)  # (B,T,out)
        return Y

class PlasticConv1d(nn.Module):
    """Plastic 1x1 Conv1d head with Hebbian/Oja fast weights."""
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 init_eta=0.1, learn_eta=False, rule="oja", bias=True):
        super().__init__()
        if int(kernel_size) != 1:
            raise ValueError("PlasticConv1d currently supports kernel_size=1 only.")
        self.in_ch = int(in_channels)
        self.out_ch = int(out_channels)
        self.W = nn.Parameter(torch.empty(out_channels, in_channels))   # 1x1 -> matrix
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.register_buffer("H", torch.zeros(1, out_channels, in_channels))
        self.update_traces = False
        self.modulators = None
        self.rule = str(rule).lower()
        self.learn_eta = bool(learn_eta)
        if learn_eta:
            self._eta_param = nn.Parameter(torch.log(torch.expm1(torch.tensor(init_eta))))
        else:
            self.register_buffer("_eta_const", torch.tensor(float(init_eta), dtype=torch.float32))

        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def reset_traces(self, batch_size: int, device=None):
        device = device or self.W.device
        self.H = torch.zeros(batch_size, self.out_ch, self.in_ch, device=device)

    def set_mode(self, *, update_traces: bool, modulators: torch.Tensor | None):
        self.update_traces = bool(update_traces)
        self.modulators = modulators

    def forward(self, x):  # x: (B, C_in, T)
        B, Cin, T = x.shape
        assert Cin == self.in_ch
        if self.H.dim() != 3 or self.H.size(0) != B:
            self.H = torch.zeros(B, self.out_ch, self.in_ch, device=x.device, dtype=x.dtype)

        eta = _eta_from_param(getattr(self, "_eta_param", None),
                              self.learn_eta, float(getattr(self, "_eta_const", 0.1)), x.device)

        outs = []
        for t in range(T):
            x_t = x[:, :, t]  # (B, Cin)
            y_static = torch.nn.functional.linear(x_t, self.W, self.bias)         # (B, Cout)
            y_fast   = torch.bmm(self.H, x_t.unsqueeze(-1)).squeeze(-1)           # (B, Cout)
            y_t = y_static + y_fast
            outs.append(y_t)

            if self.update_traces:
                if self.modulators is None:
                    m_t = torch.ones(B, device=x.device, dtype=x.dtype)
                else:
                    m = self.modulators
                    if m.dim() == 1:
                        m_t = m.expand(B)
                    elif m.dim() == 2:
                        m_t = m[:, t]
                    else:
                        raise ValueError("modulators must be (T,), (1,T), or (B,T)")
                    m_t = m_t.to(dtype=x.dtype, device=x.device)

                outer = torch.einsum("bo,bi->boi", y_t, x_t)  # (B, Cout, Cin)
                if self.rule == "hebb":
                    dH = eta * m_t.view(B, 1, 1) * outer
                else:  # oja
                    dH = eta * m_t.view(B, 1, 1) * (outer - (y_t.pow(2).unsqueeze(-1) * self.H))
                self.H = self.H + dH

        Y = torch.stack(outs, dim=-1)  # (B, Cout, T)
        return Y
