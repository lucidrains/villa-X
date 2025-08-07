import torch
from torch.nn import Module

from x_transformers import Encoder

from vit_pytorch.vit_3d import ViT as SpaceTimeViT

from vector_quantize_pytorch import FSQ

from rectified_flow_pytorch import RectifiedFlow

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# ACT latent

class LatentActionModel(Module):
    def __init___(
        self,
        space_time_vit: SpaceTimeViT,
        fsq_levels = (8, 5, 5, 5),
        fsq_num_codebooks = 2, # channel-splitting from nvidia
    ):
        super().__init__()
        self.space_time_vit = space_time_vit

        self.fsq = FSQ(
            levels = fsq_levels,
            num_codebooks = fsq_num_codebooks
        )

class ACTLatent(Module):
    def __init__(
        self,
        encoder: Encoder
    ):
        super().__init__()
        self.encoder = encoder
        self.flow = RectifiedFlow(encoder)

class ACTRobot(Module):
    def __init__(
        self,
        encoder: Encoder
    ):
        super().__init__()
        self.encoder = encoder
        self.flow = RectifiedFlow(encoder)

# the main class

class ViLLaX(Module):
    def __init__(
        self,
        lam: LatentActionModel,
        act_latent: ACTLatent,
        act_robot: ACTRobot
    ):
        super().__init__()
        self.lam = lam
        self.act_latent = act_latent
        self.act_robot = act_robot
