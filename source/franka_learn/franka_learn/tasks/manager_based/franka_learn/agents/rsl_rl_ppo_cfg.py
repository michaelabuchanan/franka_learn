# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

#from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import *

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 5000
    experiment_name = "camera_lifting"
    class_name = "OnPolicyRunner"
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_obs_normalization=False,
    #     critic_obs_normalization=False,
    #     actor_hidden_dims=[64, 64],
    #     critic_hidden_dims=[64, 64],
    #     activation="elu",
    # )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # Maps environment observation groups onto the algorithm's observation sets.
    # The model sorts each group into 1-D (MLP) or 2-D (CNN) inputs by its rank,
    # and builds one CNN encoder per 2-D group.
    obs_groups = {
        "actor": ["policy", "image_front", "image_side"],
        "critic": ["policy", "image_front", "image_side"],
    }

    actor = RslRlCNNModelCfg(
        class_name="CNNModel",
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 64, 64],
            kernel_size=3,
            stride=2,
            padding="zeros",
            activation="relu",
        ),
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic = RslRlCNNModelCfg(
        class_name="CNNModel",
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 64, 64],
            kernel_size=3,
            stride=2,
            padding="zeros",
            activation="relu",
        ),
        hidden_dims=[256, 128, 128],
        activation="elu",
    )