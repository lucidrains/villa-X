from __future__ import annotations

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

# flow DiT

# random sinusoidal for times

class RandomSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

# DiT wrapper

class FlowTransformerWrapper(Module):
    def __init__(
        self,
        dim_input,
        dim_time,
        transformer: Encoder,
        dropout_vlm_key_values = 0.5
    ):
        super().__init__()

        self.transformer = transformer

        dim = transformer.dim

        self.proj_in = nn.Linear(dim_input, dim)

        self.to_time_cond = nn.Sequential(
            RandomSinusoidalPosEmb(dim),
            nn.Linear(dim, dim_time),
            nn.SiLU(),
        )

        self.proj_out = nn.Linear(dim, dim_input)

        # there is a practice circulating around of structured dropout of vlm key values (or is it to the latents? figure out later)

        self.dropout_vlm_key_values = dropout_vlm_key_values

    def forward(
        self,
        actions,
        *,
        times,
        context = None,
        context_mask = None,
        vlm_key_values = None,
        vlm_seq_mask = None
    ):
        batch_size, device = actions.shape[0], actions.device

        time_cond = self.to_time_cond(times)

        tokens = self.proj_in(actions)

        # structured dropout by attn masking out to vlm key / values (50% in paper)

        if exists(vlm_key_values):
            assert exists(vlm_seq_mask)
            vlm_kv_dropout = torch.rand(batch_size, device = device) < self.dropout_vlm_key_values
            vlm_seq_mask = einx.logical_and('b, b n -> b n', vlm_kv_dropout, vlm_seq_mask)

        attended = self.transformer(
            tokens,
            condition = condition,
            context = context,
            context_mask = context_mask,
            self_attn_additional_kv = vlm_key_values,
            detach_additional_kv = True,
            additional_kv_mask = vlm_seq_mask
        )

        pred = self.proj_out(attended)
        return pred

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
        flow_dit: FlowTransformerWrapper
    ):
        super().__init__()
        self.flow_dit = flow_dit
        self.flow_wrapper = RectifiedFlow(flow_dit)

class ACTRobot(Module):
    def __init__(
        self,
        flow_dit: FlowTransformerWrapper
    ):
        super().__init__()
        self.flow_dit = flow_dit
        self.flow_wrapper = RectifiedFlow(flow_dit)

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
