from __future__ import annotations

import torch
from torch import nn, cat, Tensor
import torch.nn.functional as F
from torch.nn import Module

from x_transformers import (
    Encoder,
    Decoder,
    AttentionPool,
    CrossAttender
)

from vit_pytorch import ViT
from vit_pytorch.extractor import Extractor

from vector_quantize_pytorch import FSQ

from rectified_flow_pytorch import RectifiedFlow

from torchvision.models import resnet18, ResNet18_Weights

import einx
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from transformers import AutoModelForVision2Seq, AutoProcessor

from axial_positional_embedding import ContinuousAxialPositionalEmbedding

# constants

KeyValues = list[tuple[Tensor, Tensor]]

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse_fn(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse_fn

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
    def __init__(
        self,
        dim,
        vit: ViT,
        dim_proprio,
        dim_actions,
        channels = 3,
        patch_size = 32,
        dim_image_model = 512,
        idm_depth = 2,
        fdm_depth = 2,
        proprio_fdm_depth = 2,
        fsq_levels = [8, 5, 5, 5],
        recon_vit_kwargs: dict = dict(
            depth = 4,
        ),
        idm_kwargs: dict = dict(),
        fdm_kwargs: dict = dict(),
        proprio_fdm_kwargs: dict = dict(),
        fsq_num_codebooks = 2, # channel-splitting from nvidia
    ):
        super().__init__()
        self.dim = dim

        self.vit = Extractor(vit, return_embeddings_only = True)

        self.to_observation_tokens = AttentionPool(dim = dim, dim_context = dim_image_model)

        self.inverse_dynamic_model = Decoder(dim = dim, depth = idm_depth, **idm_kwargs)

        self.forward_dynamic_model = Decoder(dim = dim, depth = fdm_depth, **fdm_kwargs)

        self.proprio_forward_dynamic_model = Decoder(dim = dim, depth = proprio_fdm_depth, **proprio_fdm_kwargs)

        self.to_proprio_fdm_tokens = nn.Linear(dim_proprio + dim, dim)

        self.to_pred_proprio = nn.Linear(dim, dim_proprio)
        self.to_pred_actions = nn.Linear(dim, dim_actions)

        self.fsq = FSQ(
            dim = dim,
            levels = fsq_levels,
            num_codebooks = fsq_num_codebooks
        )

        self.recon_spatial_queries = ContinuousAxialPositionalEmbedding(
            dim = dim_image_model,
            num_axial_dims = 2
        )

        self.recon_vit = Encoder(dim = dim_image_model, **recon_vit_kwargs)

        self.patch_size = patch_size

        self.to_recon_pred = nn.Sequential(
            nn.Linear(dim_image_model, channels * patch_size ** 2),
            Rearrange('b h w (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

    def forward(
        self,
        video,          # (b c t h w)
        proprio = None, # (b t dp)
        actions = None  # (b t da)
    ):
        batch, time = video.shape[0], video.shape[2]

        return_loss = exists(proprio) and exists(actions)

        images = rearrange(video, 'b c t h w -> b t c h w')

        images, inverse_pack_time = pack_with_inverse(images, '* c h w')

        embeddings = self.vit(images) # b n d

        assert embeddings.shape[-1] == self.dim

        observe_tokens = self.to_observation_tokens(embeddings)

        observe_tokens = rearrange(observe_tokens, 'bt 1 d -> bt d')
        observe_tokens = inverse_pack_time(observe_tokens, '* d') # b t d

        latent_actions = self.inverse_dynamic_model(observe_tokens)

        quantized_latent_actions, _ = self.fsq(latent_actions)

        if not return_loss:
            return quantized_latent_actions[:, 1:]

        latent_actions = latent_actions[:, 1:]
        quantized_latent_actions = quantized_latent_actions[:, 1:]

        recon_observe_tokens = self.forward_dynamic_model(quantized_latent_actions)

        video_height_patches, video_width_patches = (video.shape[-2] // self.patch_size), (video.shape[-1] // self.patch_size)
        detr_queries = self.recon_spatial_queries((video_height_patches, video_width_patches))

        detr_queries = repeat(detr_queries, '... -> bt ...', bt = batch * (time - 1))
        recon_observe_tokens = rearrange(recon_observe_tokens, 'b t ... -> (b t) ...')

        recon_tokens, packed_shape = pack((recon_observe_tokens, detr_queries), 'b * d')

        attended_recon_tokens = self.recon_vit(recon_tokens)

        attended_recon_tokens = attended_recon_tokens[:, 1:] # keep only detr queries

        # assume square

        attended_recon_tokens = rearrange(attended_recon_tokens, 'b (h w) ... -> b h w ...', h = video_height_patches)

        recon_video = self.to_recon_pred(attended_recon_tokens)

        ar_fdm_loss = F.l1_loss(rearrange(video[:, :, 1:], 'b c t h w -> (b t) c h w'), recon_video)

        proprio, proprio_target = proprio[:, :-1], proprio[:, 1:]
        actions_target = actions[:, 1:]

        proprio_fdm_tokens = self.to_proprio_fdm_tokens(cat((quantized_latent_actions, proprio), dim = -1))

        proprio_fdm_embed = self.proprio_forward_dynamic_model(quantized_latent_actions)

        pred_proprio = self.to_pred_proprio(proprio_fdm_embed)
        pred_action = self.to_pred_actions(proprio_fdm_embed)

        ar_proprio_loss = F.l1_loss(pred_proprio, proprio_target)
        ar_action_loss = F.l1_loss(pred_action, actions_target)

        # losses

        total_loss = (
            ar_fdm_loss + 
            ar_proprio_loss +
            ar_action_loss
        )

        loss_breakdown = (ar_fdm_loss, ar_proprio_loss, ar_action_loss)

        return total_loss, loss_breakdown

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

    def forward(
        self,
        vlm_key_values: KeyValues,
        wrist_image = None
    ):
        sampled_action_latents = self.act_latent.sample(vlm_key_values = vlm_key_values)

        sampled_actions = self.act_robot.sample(
            sampled_action_latents,
            vlm_key_values = vlm_key_values,
            wrist_image = wrist_image
        )

        return sampled_actions
