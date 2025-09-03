import math
import torch
import torch.nn as nn

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from terratorch import BACKBONE_REGISTRY

from mod_res_damage.models.utils import DoubleConv2d


class TerraMindEncoder(nn.Module):
    def __init__(self,
                 version: int = "v1_base",
                 pretrained: bool = True,
                 finetune: bool = False,
                 modalities: list[str] = ['S1RTC'],
                 output_layers: list[str] = [5, 7, 9, 11]):
        super().__init__()
        self.pretrained = pretrained
        self.finetune = finetune
        self.version = version
        self.output_layers = output_layers
        
        self.terramind = BACKBONE_REGISTRY.build(f'terramind_{version}', 
                                                 pretrained=pretrained,
                                                 modalities=modalities)
        self.embed_dim = 768 if version == "v1_base" else 1024
        if not self.finetune:
            for param in self.terramind.parameters():
                param.requires_grad = False
        

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for mod, tensor in x.items():
            mod_mapped = self.terramind.mod_name_mapping[mod]
            mod_dict = self.terramind.encoder_embeddings[mod_mapped](tensor)
            embeddings.append(mod_dict['x'] + mod_dict['emb'])
        embeddings = torch.cat(embeddings, dim=1)
        
        out = []
        for i, block in enumerate(self.terramind.encoder):
            embeddings = block(embeddings)
            if i in self.output_layers:
                out.append(embeddings.clone())
        return out
      

class TerramindBayeSiamNet(TerraMindEncoder):
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
                 num_classes: int,
                 prior_parameters: dict = None,
                 img_size: int = 256,
                 bilinear: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.prior_parameters = prior_parameters if prior_parameters is not None else self._default_prior_parameters
        self.bilinear = bilinear
        
        self.bayes_up_layers = nn.ModuleList()
        self.bayes_ij_layers = nn.ModuleList()
        self.bayes_dc_layers = nn.ModuleList()
        for i in range(len(self.output_layers) - 1):
            dim0 = self.embed_dim // (2 ** i)
            dim1 = self.embed_dim // (2 ** (i + 1))

            if bilinear:
                up_dnn = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       DoubleConv2d(in_channels=dim0, out_channels=dim1))
            else:
                up_dnn = nn.Sequential(nn.ConvTranspose2d(dim0, dim1, kernel_size=2, stride=2),
                                       DoubleConv2d(in_channels=dim1, out_channels=dim1))
            dnn_to_bnn(up_dnn, self.prior_parameters)
            self.bayes_up_layers.append(up_dnn)

            ij_scale = 2 ** (i + 1)
            if bilinear:
                ij_dnn = nn.Sequential(nn.Upsample(scale_factor=ij_scale, mode='bilinear', align_corners=True),
                                       DoubleConv2d(in_channels=self.embed_dim, out_channels=dim1))
            else:
                ij_dnn = nn.Sequential(nn.ConvTranspose2d(self.embed_dim, dim1, kernel_size=ij_scale, stride=ij_scale),
                                       DoubleConv2d(in_channels=dim1, out_channels=dim1))
            dnn_to_bnn(ij_dnn, self.prior_parameters)
            self.bayes_ij_layers.append(ij_dnn)

            dc_dnn = DoubleConv2d(in_channels=dim1*2, out_channels=dim1)
            dnn_to_bnn(dc_dnn, self.prior_parameters)
            self.bayes_dc_layers.append(dc_dnn)

        dim = self.embed_dim // (2 ** (len(self.output_layers) - 1))
        self.final_conv = nn.Conv2d(in_channels=dim, out_channels=num_classes, kernel_size=1)
        
        self.img_size = img_size
        self.output_layers = self.output_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x['S1RTC']
        _, _, t, _, _ = input.shape
        
        feat_before = self.forward_encoder({"S1RTC": input[:, : , :t//2].squeeze(2)})
        feat_after = self.forward_encoder({"S1RTC": input[:, : , t//2: ].squeeze(2)})
        b, l, d = feat_after[0].shape
        v = math.floor(l ** 0.5)
        
        diffs = []
        for before, after in zip(feat_before, feat_after):
            diff = after - before
            diff = diff.transpose(1, 2).reshape(b, d, v, v)
            diffs.append(diff)
            
        for i in range(len(self.output_layers) - 1):
            if i == 0:
                d = diffs[i]

            d_up = self.bayes_up_layers[i](d)
            d_pr = self.bayes_ij_layers[i](diffs[i+1])
            d = torch.cat([d_up, d_pr], dim=1)
            d = self.bayes_dc_layers[i](d)
        
        d = self.final_conv(d)    
        
        return d
