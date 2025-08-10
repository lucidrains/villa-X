import pytest

import torch

def test_villa_x():
    from villa_x import ACTLatent, ACTRobot

    act_latent = ACTLatent()

    act_robot = ACTRobot()

    # training

    action_latents = torch.randn(1, 32, 128)
    loss = act_latent(action_latents)
    loss.backward()

    actions = torch.randn(1, 128, 20)
    loss = act_robot(actions, action_latents)
    loss.backward()

    # hierarchical inference

    sampled_action_latents = act_latent.sample()

    sampled_actions = act_robot.sample(sampled_action_latents)

    assert sampled_actions.shape == (1, 128, 20)
