import pytest
param = pytest.mark.parametrize

import torch

@param('send_vlm_key_values', (False, True))
def test_villa_x(
    send_vlm_key_values
):
    from villa_x import ACTLatent, ACTRobot

    act_latent = ACTLatent()

    act_robot = ACTRobot()

    # vlm key values

    vlm_kv = None

    if send_vlm_key_values:
        # say top 2 layers

        vlm_kv = [
            (torch.randn(1, 4, 32, 64), torch.randn(1, 4, 32, 64)),
            (torch.randn(1, 4, 32, 64), torch.randn(1, 4, 32, 64))
        ]

    # training

    action_latents = torch.randn(1, 32, 128)
    loss = act_latent(action_latents, vlm_key_values = vlm_kv)
    loss.backward()

    actions = torch.randn(1, 128, 20)
    loss = act_robot(actions, action_latents,vlm_key_values = vlm_kv)
    loss.backward()

    # hierarchical inference

    sampled_action_latents = act_latent.sample()

    sampled_actions = act_robot.sample(sampled_action_latents, vlm_key_values = vlm_kv)

    assert sampled_actions.shape == (1, 128, 20)
