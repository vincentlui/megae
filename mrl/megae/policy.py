import mrl
from mrl.utils.misc import soft_update, flatten_state
from mrl.utils.networks import StochasticActor

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
        self.action_scale = self.env.max_action

    def __call__(self, state, greedy=False, context=None, is_explore=None):
        # initial exploration and intrinsic curiosity
        res = None
        if self.training:
            if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
                res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
            elif hasattr(self, 'ag_curiosity'):
                state = self.ag_curiosity.relabel_state(state)

        if res is not None:
            return res

        s_flatten = flatten_state(state)
        action = None
        if is_explore is not None and is_explore.any():
            num_env = s_flatten.shape[0]
            action_dim = self.env.action_dim
            action = np.empty((num_env, action_dim))

            is_explore_index = np.nonzero(is_explore)[0]
            goal_index = np.nonzero(is_explore==0)[0]
            if len(is_explore_index) > 0:
                if isinstance(state, dict):
                    obs = state['observation']
                    s = np.concatenate((obs[is_explore_index], context[is_explore_index]), -1)
                else:
                    s = np.concatenate((state[is_explore_index], context[is_explore_index]), -1)
                a_expl = self.directed_exploration(s, greedy)
                action[is_explore_index] = a_expl
            if len(goal_index) > 0:
                a_goal = self.goal_conditioned_policy(s_flatten[goal_index], greedy)
                action[goal_index] = a_goal
        else:
            action = self.goal_conditioned_policy(s_flatten, greedy)

        return np.clip(action, -self.action_scale, self.action_scale)

    def goal_conditioned_policy(self, state, greedy):
        if hasattr(self, 'state_normalizer'):
            state = self.state_normalizer(state, update=self.training)

        state = self.torch(state)

        if self.use_actor_target:
            action = self.actor_target(state)
        else:
            action = self.actor(state)

        if isinstance(self.actor.model, StochasticActor):
            action = action[0]

        action = self.numpy(action)

        if isinstance(self.actor.model, StochasticActor) and self.training and not greedy:
            if not isinstance(self.actor.model, StochasticActor):
                action = self.action_noise(action)

            if self.config.get('eexplore'):
                eexplore = self.config.eexplore
                if hasattr(self, 'ag_curiosity'):
                    eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
                mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
                randoms = np.random.random(action.shape) * (2 * self.action_scale) - self.action_scale
                action = mask * randoms + (1 - mask) * action

        return action

    def directed_exploration(self, state, greedy):
        # Directed exploration
        if hasattr(self, 'state_normalizer_expl'):
            state = self.state_normalizer_expl(state, update=self.training)

        state = self.torch(state)

        if self.use_actor_target:
            action = self.expl_actor_target(state)
        else:
            action = self.expl_actor(state)

        if isinstance(self.expl_actor.model, StochasticActor):
            action = action[0]

        action = self.numpy(action)

        if isinstance(self.actor.model, StochasticActor) and self.training and not greedy:
            if not isinstance(self.actor.model, StochasticActor):
                action = self.action_noise(action)

            if self.config.get('eexplore'):
                eexplore = self.config.eexplore
                if hasattr(self, 'ag_curiosity'):
                    eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
                mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
                randoms = np.random.random(action.shape) * (2 * self.action_scale) - self.action_scale
                action = mask * randoms + (1 - mask) * action

        return action


class StochasticActorPolicy(mrl.Module):
    """Used for SAC / learned action noise"""

    def __init__(self):
        super().__init__(
            'policy',
            required_agent_modules=[
                'actor', 'env', 'replay_buffer'
            ],
            locals=locals())

    def _setup(self):
        self.use_actor_target = self.config.get('use_actor_target')

    def __call__(self, state, greedy=False):
        action_scale = self.env.max_action

        # initial exploration and intrinsic curiosity
        res = None
        if self.training:
            if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
                res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
            elif hasattr(self, 'ag_curiosity'):
                state = self.ag_curiosity.relabel_state(state)

        state = flatten_state(state)  # flatten goal environments
        if hasattr(self, 'state_normalizer'):
            state = self.state_normalizer(state, update=self.training)

        if res is not None:
            return res

        state = self.torch(state)

        if self.use_actor_target:
            action, _ = self.actor_target(state)
        else:
            action, _ = self.actor(state)
        action = self.numpy(action)

        if self.training and not greedy and self.config.get('eexplore'):
            eexplore = self.config.eexplore
            if hasattr(self, 'ag_curiosity'):
                eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
            mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
            randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
            action = mask * randoms + (1 - mask) * action

        return np.clip(action, -action_scale, action_scale)