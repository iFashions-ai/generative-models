from typing import Callable, Dict
import torch
import torch.nn as nn
from packaging import version
from collections import OrderedDict

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

        self.patches: Dict[str, OrderedDict[str, Callable]] = {
            "input_block": OrderedDict()
        }


    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, patches=self.patches, **kwargs)

    def add_patch(self, fn: Callable[[torch.Tensor, int], torch.Tensor], patch_location: str, patch_name: str):
        if patch_location not in self.patches:
            raise ValueError(f"Unknown patch location {patch_location}")
        self.patches[patch_location][patch_name] = fn



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
