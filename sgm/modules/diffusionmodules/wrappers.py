from typing import Callable
import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

        self.patches = {
            "input_block": []
        }


    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, patches=self.patches, **kwargs)

    def add_patch(self, fn: Callable[[torch.Tensor, int], torch.Tensor], name):
        if name not in self.patches:
            raise ValueError(f"Unknown patch name {name}")
        self.patches[name].append(fn)



class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            patches=self.patches,
            **kwargs,
        )
