import torch
import wandb

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from mod_res_damage.models.prbm import ParllelRBMs
from mod_res_damage.datasets.mbright import ModifiedBright


def pretrain(prbm, data, batch_size, epochs, lr):
    wandb.init(
        project="mod_res_damage",
        name="init_prbm_64_time_01",
        dir="/ceoas/Vandenhoek_Lab/mod_res_damage/data/prbm",
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for j, batch in enumerate(loader):
            inputs, _, _ = batch["inputs"]["S1GRD"], batch["target"], batch["no_data"]
            inputs = inputs.to(torch.device('cuda:1'))
            B, C, T, H, W = inputs.shape
            inputs = inputs.permute(0, 3, 4, 1, 2).reshape(-1, C, T)
            mask = (inputs != 0.0).any(dim=(1, 2))
            inputs = inputs[mask]
            for i in range(len(prbm.rbms)):
                if i > 0:
                    inputs = inputs[:,:,1:] - inputs[:,:,:-1]
                    L, C, T = inputs.shape
                flat_inputs = inputs.reshape(-1, C*T)  # (B*H*W, C*T)
                fe_data, fe_model = prbm.rbms[i].contrastive_divergence(flat_inputs, lr)
                wandb.log({f"batch_fe_data_rbm_{i}": fe_data})
                wandb.log({f"batch_fe_model_rbm_{i}": fe_model})
    out_path = Path("/ceoas/Vandenhoek_Lab/mod_res_damage/data/prbm")
    out_path.mkdir(exist_ok=True)
    torch.save(prbm.state_dict(), out_path / "init_time_01.pt")
    wandb.finish()


def main():
    prbm = ParllelRBMs([[20, 64],
                        [18, 64],
                        [16, 64]], k=4)
    prbm.to("cuda:1")
    data = ModifiedBright(split="val",
        root_path="./data/mbright_gee",
        input_size=224,
        average_time=False,
        augment=None,
        holdout = None,
        exclude = None)
    pretrain(prbm, data, batch_size=1, epochs=2048, lr=0.01)


if __name__ == "__main__":
    main()