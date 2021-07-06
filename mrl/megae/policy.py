import mrl
from mrl.utils.misc import soft_update, flatten_state
from mrl.modules.model import PytorchModel

import numpy as np
import torch
import torch.nn.functional as F
import os


class ExplorationActorPolicy(mrl.Module):
    """Used for DDPG / TD3 and other deterministic policy variants"""

    def __init__(self):
        super().__init__(
            'policy',
            required_agent_modules=[
                'actor', 'action_noise', 'env', 'replay_buffer'
            ],
            locals=locals())

    def _setup(self):
        self.use_actor_target = self.config.get('use_actor_target')

    def __call__(self, state, context=None, greedy=False):
        action_scale = self.env.max_action

        # initial exploration and intrinsic curiosity
        res = None
        if self.training:
            if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
                res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
            elif hasattr(self, 'ag_curiosity'):
                state = self.ag_curiosity.relabel_state(state)

        if res is not None:
            return res

        if context is None:
            # Goal conditioned exploration
            state = flatten_state(state)  # flatten goal environments
            if hasattr(self, 'state_normalizer'):
                state = self.state_normalizer(state, update=self.training)

            state = self.torch(state)

            if self.use_actor_target:
                action = self.numpy(self.actor_target(state))
            else:
                action = self.numpy(self.actor(state))

            if self.training and not greedy:
                action = self.action_noise(action)
                if self.config.get('eexplore'):
                    eexplore = self.config.eexplore
                    if hasattr(self, 'ag_curiosity'):
                        eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
                    mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
                    randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
                    action = mask * randoms + (1 - mask) * action

        else:
            # Directed exploration
            if isinstance(state, dict):
                obs = state['observation']
                state = np.concatenate((obs, context), -1)
            else:
                state = np.concatenate((state, context), -1)

            if hasattr(self, 'state_normalizer_expl'):
                state = self.state_normalizer_expl(state, update=self.training)

            state = self.torch(state)

            if self.use_actor_target:
                action = self.numpy(self.expl_actor_target(state))
            else:
                action = self.numpy(self.expl_actor(state))

            if self.training and not greedy:
                action = self.action_noise(action)


        return np.clip(action, -action_scale, action_scale)


# class StochasticActorPolicy(mrl.Module):
#     """Used for SAC / learned action noise"""
#
#     def __init__(self):
#         super().__init__(
#             'policy',
#             required_agent_modules=[
#                 'actor', 'env', 'replay_buffer'
#             ],
#             locals=locals())
#
#     def _setup(self):
#         self.use_actor_target = self.config.get('use_actor_target')
#
#     def __call__(self, state, greedy=False):
#         action_scale = self.env.max_action
#
#         # initial exploration and intrinsic curiosity
#         res = None
#         if self.training:
#             if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
#                 res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
#             elif hasattr(self, 'ag_curiosity'):
#                 state = self.ag_curiosity.relabel_state(state)
#
#         state = flatten_state(state)  # flatten goal environments
#         if hasattr(self, 'state_normalizer'):
#             state = self.state_normalizer(state, update=self.training)
#
#         if res is not None:
#             return res
#
#         state = self.torch(state)
#
#         if self.use_actor_target:
#             action, _ = self.actor_target(state)
#         else:
#             action, _ = self.actor(state)
#         action = self.numpy(action)
#
#         if self.training and not greedy and self.config.get('eexplore'):
#             eexplore = self.config.eexplore
#             if hasattr(self, 'ag_curiosity'):
#                 eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
#             mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
#             randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
#             action = mask * randoms + (1 - mask) * action
#
#         return np.clip(action, -action_scale, action_scale)