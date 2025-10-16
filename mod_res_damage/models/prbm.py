import torch
import torch.nn as nn
import torch.nn.functional as F 

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from torch.utils.data import DataLoader, TensorDataset

from mod_res_damage.models.utils import DoubleConv2d


class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=4, gaussian=True):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.gaussian = gaussian

        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        if self.gaussian:
            prob_h = F.linear(v, self.W, self.h_bias)
            h_sample = prob_h + torch.randn_like(prob_h)
            return prob_h, h_sample
        else:
            prob_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
            return prob_h, torch.bernoulli(prob_h)
        
    def sample_v(self, h):
        if self.gaussian:
            prob_v = F.linear(h, self.W.t(), self.v_bias)
            v_sample = prob_v + torch.randn_like(prob_v)
            return prob_v, v_sample
        else:
            prob_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
            return prob_v, torch.bernoulli(prob_v)

    def forward(self, v):
        prob_h, h_sample = self.sample_h(v)
        for _ in range(self.k):
            prob_v, v_sample = self.sample_v(h_sample)
            prob_h, h_sample = self.sample_h(v_sample)
        return v, v_sample

    def free_energy(self, v):
        # v: (batch, n_visible)
        vbias_term = 0.5 * torch.sum((v - self.v_bias)**2, dim=1)            # 1/2 ||v-a||^2
        hidden_mean = F.linear(v, self.W, self.h_bias)                       # W^T v + b  if W is (n_v, n_h)
        hidden_term = -0.5 * torch.sum(hidden_mean**2, dim=1)               # -1/2 ||W^T v + b||^2
        return vbias_term + hidden_term                                     # shape (batch,)

    def contrastive_divergence(self, v, lr):
        # Positive phase
        prob_h0, h0 = self.sample_h(v)

        # Negative phase (k-step Gibbs)
        hk = h0
        for _ in range(self.k):
            prob_vk, vk = self.sample_v(hk)
            prob_hk, hk = self.sample_h(vk)

        # Parameter updates
        self.W.data += lr * (torch.matmul(prob_h0.t(), v) - torch.matmul(prob_hk.t(), vk)) / v.size(0)
        self.v_bias.data += lr * torch.mean(v - vk, dim=0)
        self.h_bias.data += lr * torch.mean(prob_h0 - prob_hk, dim=0)

        # Free energies
        fe_data = self.free_energy(v).mean().item()
        fe_model = self.free_energy(vk).mean().item()
        return fe_data, fe_model

        
class ParllelRBMs(nn.Module):
    def __init__(self, layer_sizes, k=4):
        super().__init__()
        self.rbms = nn.ModuleList([RBM(layer_sizes[i][0], layer_sizes[i][1], k=k) for i in range(len(layer_sizes))])
        self.layer_sizes = layer_sizes

    def forward(self, x):
        outputs = []
        inputs = x

        B, C, T, H, W = inputs.shape
        inputs = inputs.permute(0, 3, 4, 1, 2).reshape(-1, C, T)
        for i in range(len(self.rbms)):
            if i > 0:
                inputs = inputs[:,:,1:] - inputs[:,:,:-1]
                L, C, T = inputs.shape
            flat_inputs = inputs.reshape(-1, C*T)
            r_out, _ = self.rbms[i].sample_h(flat_inputs)
            r_out = r_out.reshape(B, H, W, self.layer_sizes[i][1]).permute(0, 3, 1, 2)
            outputs.append(r_out)
            
        return torch.concatenate(outputs, dim=1)
    

class ParllelRBMsBayesRegressor(nn.Module):
    _default_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.5,
    }
    
    def __init__(self,
                 num_predictands: int,
                 prior_parameters: dict = None,
                 img_size: int = 224,
                 layer_sizes: list[list[int]]= [[20, 64],[18, 64],[16, 64]],
                 state_path: str | None = None,
                 finetune: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.prbms = ParllelRBMs(layer_sizes)
        if state_path is not None:
            state_dict = torch.load(state_path)
            self.prbms.load_state_dict(state_dict)
        
        self.finetune = finetune
        if not self.finetune:
            for param in self.prbms.parameters():
                param.requires_grad = False
        
        self.prior_parameters = prior_parameters if prior_parameters is not None else self._default_prior_parameters

        dim = sum([h for _,h in layer_sizes])
        self.final_conv = DoubleConv2d(in_channels=dim, out_channels=num_predictands)
        dnn_to_bnn(self.final_conv, self.prior_parameters)
        
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x['S1GRD']
        _, _, t, _, _ = inputs.shape
        
        l = self.prbms(inputs)        
        p = self.final_conv(l)    
        
        return p