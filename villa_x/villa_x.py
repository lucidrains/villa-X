from __future__ import annotations

import torch
from torch import nn, cat, Tensor
from torch.nn import Module

from x_transformers import Encoder

from vit_pytorch.vit_3d import ViT as SpaceTimeViT

from vector_quantize_pytorch import FSQ

from rectified_flow_pytorch import RectifiedFlow

from torchvision.models import resnet18, ResNet18_Weights

import einx
from einops import rearrange, pack
from einops.layers.torch import Rearrange

from transformers import AutoModelForVision2Seq, AutoProcessor

# constants

KeyValues = list[tuple[Tensor, Tensor]]

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# vlm

class VLM(Module):
    def __init__(
        self,
        model_name = 'google/paligemma-3b-pt-224'
    ):
        super().__init__()

        self.vlm = AutoModelForVision2Seq.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(
        self,
        images: Tensor,
        commands: list[str]

    ) -> KeyValues:

        # extract the cached key / values
        raise NotImplementedError

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
        dim_time = 512,
        transformer: Encoder | dict = dict(
            dim = 512,
            depth = 6,
            heads = 8,
            attn_dim_head = 64,
        ),
        cross_attend = False,
        cross_attn_dim_context = 128,
        dropout_vlm_key_values = 0.5
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Encoder(
                dim_condition = dim_time,
                use_adaptive_layernorm = True,
                use_adaptive_layerscale = True,
                cross_attend = cross_attend,
                cross_attn_dim_context = cross_attn_dim_context,
                **transformer
            )

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
        vlm_key_values: KeyValues | None = None,
        vlm_seq_mask = None,
        prepend_tokens = None
    ):
        batch_size, device = actions.shape[0], actions.device

        time_cond = self.to_time_cond(times)

        tokens = self.proj_in(actions)

        # maybe prepend embeds

        if exists(prepend_tokens):
            prepend_len = prepend_tokens.shape[1]
            tokens = cat((prepend_tokens, tokens), dim = 1)

        # structured dropout by attn masking out to vlm key / values (50% in paper)

        if self.training and exists(vlm_key_values) and len(vlm_key_values) > 0:

            if not exists(vlm_seq_mask):
                vlm_seq_len = vlm_key_values[0][0].shape[-2]
                vlm_seq_mask = torch.ones((batch_size, vlm_seq_len), device = device)

            vlm_kv_dropout = torch.rand(batch_size, device = device) < self.dropout_vlm_key_values
            vlm_seq_mask = einx.logical_and('b, b n -> b n', vlm_kv_dropout, vlm_seq_mask)

        attended = self.transformer(
            tokens,
            condition = time_cond,
            context = context,
            context_mask = context_mask,
            self_attn_additional_kv = vlm_key_values,
            detach_additional_kv = True,
            additional_kv_mask = vlm_seq_mask
        )

        if exists(prepend_tokens):
            attended = attended[:, prepend_len:]

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
        flow_dit: dict | FlowTransformerWrapper = dict(
            dim_input = 128
        )
    ):
        super().__init__()

        if isinstance(flow_dit, dict):
            flow_dit = FlowTransformerWrapper(**flow_dit)

        self.flow_dit = flow_dit
        self.flow_wrapper = RectifiedFlow(flow_dit)

    def sample(
        self,
        *args,
        **kwargs
    ):
        return self.flow_wrapper.sample(*args, **kwargs)

    def forward(
        self,
        action_latents,
        **kwargs
    ):
        return self.flow_wrapper(action_latents, **kwargs)

class ACTRobot(Module):
    def __init__(
        self,
        dim_proprio = None,
        dim_action_latent = 128,
        flow_dit: dict | FlowTransformerWrapper = dict(
            dim_input = 20
        )
    ):
        super().__init__()

        if isinstance(flow_dit, dict):
            flow_dit = FlowTransformerWrapper(
                cross_attend = True,
                cross_attn_dim_context = dim_action_latent,
                **flow_dit
            )

        self.flow_dit = flow_dit
        self.flow_wrapper = RectifiedFlow(flow_dit)

        dim_model = flow_dit.transformer.dim

        # take care of wrist image tokens
        # only provided for ACT-Robot for some reason
    
        weights = ResNet18_Weights.DEFAULT

        self.wrist_image_transform = weights.transforms()

        self.wrist_encoder = resnet18(weights = weights, progress = False)

        self.wrist_encoder.avgpool = nn.Identity()

        self.wrist_encoder.fc = Rearrange('b (c n) -> b n c', c = 512)

        self.wrist_feats_to_encoded = nn.Linear(512, dim_model)

        # proprio token at time t

        self.encode_proprio = nn.Linear(dim_proprio, dim_model) if exists(dim_proprio) else None

    def encode_wrist_state(
        self,
        image
    ):
        transformed = self.wrist_image_transform(image)
        wrist_feats = self.wrist_encoder(transformed)
        return self.wrist_feats_to_encoded(wrist_feats)

    def sample(
        self,
        action_latents,
        *args,
        wrist_image = None,
        **kwargs
    ):

        prepend_tokens = None

        if exists(wrist_image):
            prepend_tokens = self.encode_wrist_state(wrist_image)

        return self.flow_wrapper.sample(*args, context = action_latents, prepend_tokens = prepend_tokens, **kwargs)

    def forward(
        self,
        actions,
        action_latents,
        *,
        wrist_image = None, # (b c h w)
        proprio = None,
        **kwargs
    ):
        prepend_tokens = []

        if exists(wrist_image):
            wrist_tokens = self.encode_wrist_state(wrist_image)
            prepend_tokens.append(wrist_tokens)

        if exists(proprio):
            assert exists(self.encode_proprio), '`dim_proprio` must be set on init to accept proprioception input'

            proprio_token = self.encode_proprio(proprio)
            prepend_tokens.append(proprio_token)

        prepend_tokens_to_dit = None

        if len(prepend_tokens) > 0:
            prepend_tokens_to_dit, _ = pack(prepend_tokens, 'b * d')

        return self.flow_wrapper(actions, context = action_latents, prepend_tokens = prepend_tokens_to_dit, **kwargs)

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
