import pytest
param = pytest.mark.parametrize

import torch

def test_lam():
    from villa_x import LatentActionModel
    from vit_pytorch import ViT

    lam = LatentActionModel(
        dim = 512,
        dim_proprio = 17,
        dim_actions = 20,
        patch_size = 32,
        vit = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 512,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        ),
    )

    video = torch.randn(2, 3, 8, 256, 256)
    proprio = torch.randn(2, 8, 17)
    actions = torch.randn(2, 8, 20)

    loss, breakdown = lam(video, proprio, actions)
    loss.backward()

    latent_action_tokens = lam(video)

    assert latent_action_tokens.shape == (2, 8 - 1, 512)

@param('send_vlm_key_values', (False, True))
@param('send_wrist_image', (False, True))
def test_villa_x(
    send_vlm_key_values,
    send_wrist_image
):
    from villa_x import (
        LatentActionModel,
        ACTLatent,
        ACTRobot,
        ViLLaX
    )

    act_latent = ACTLatent()

    act_robot = ACTRobot(dim_proprio = 11)

    # vlm key values

    vlm_kv = None

    if send_vlm_key_values:
        # say top 2 layers

        vlm_kv = [
            (torch.randn(1, 4, 32, 64), torch.randn(1, 4, 32, 64)),
            (torch.randn(1, 4, 32, 64), torch.randn(1, 4, 32, 64))
        ]

    # maybe wrist image

    wrist_image = None
    if send_wrist_image:
        wrist_image = torch.randn(1, 3, 224, 224)

    # training

    action_latents = torch.randn(1, 32, 128)
    loss = act_latent(action_latents, vlm_key_values = vlm_kv)
    loss.backward()

    actions = torch.randn(1, 128, 20)
    proprio = torch.randn(1, 11)

    loss = act_robot(actions, action_latents, proprio = proprio, vlm_key_values = vlm_kv, wrist_image = wrist_image)
    loss.backward()

    # hierarchical inference

    villa_x = ViLLaX(
        lam = LatentActionModel(
            dim = 512,
            dim_proprio = 17,
            dim_actions = 20,
            patch_size = 32,
            vit = dict(
                image_size = 256,
                patch_size = 32,
                num_classes = 1000,
                dim = 512,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            ),
        ),
        act_latent = act_latent,
        act_robot = act_robot
    )

    sampled_actions = villa_x(vlm_key_values = vlm_kv)

    assert sampled_actions.shape == (1, 128, 20)
