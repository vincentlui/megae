import mrl
from mrl.utils.misc import soft_update, flatten_state
from mrl.modules.model import PytorchModel
from mrl.algorithms.continuous_off_policy import OffPolicyActorCritic

import numpy as np
import torch
import torch.nn.functional as F
import os

class OffPolicyActorCritic2(OffPolicyActorCritic):
    def __init__(self, algorithm_name, optimize_every=1, actor_name='actor', critic_name='critic', replay_buffer_name='replay_buffer',
                 is_explore=False, **kwargs):
        mrl.Module.__init__(
        self,
        algorithm_name,
        required_agent_modules=['actor','critic','replay_buffer', 'env'],
        locals=locals())
        self.step = 0
        self.optimize_every = optimize_every
        self.actor_name = actor_name
        self.critic_name = critic_name
        self.replay_buffer_name = replay_buffer_name
        self.is_explore = is_explore
        self.clip_target_range = kwargs['clip_target_range']
        self.target_network_update_freq = kwargs['target_network_update_freq']
        self.target_network_update_frac = kwargs['target_network_update_frac']
        self.action_l2_regularization = kwargs['action_l2_regularization']
        self.actor_weight_decay = kwargs['actor_weight_decay']
        self.critic_weight_decay = kwargs['critic_weight_decay']
        self.kwargs = kwargs

    def _setup(self):
        """Sets up actor/critic optimizers and creates target network modules"""

        self.targets_and_models = []

        # Actor setup
        actor_params = []
        self.actors = []
        for module in list(self.module_dict.values()):
            name = module.module_name
            if name.startswith(self.actor_name) and isinstance(module, PytorchModel):
                self.actors.append(module)
                actor_params += list(module.model.parameters())
                target = module.copy(name + '_target')
                target.model.load_state_dict(module.model.state_dict())
                # Freeze target networks with respect to optimizers (only update via polyak averaging)
                for p in target.model.parameters():
                    p.requires_grad = False
                self.agent.set_module(name + '_target', target)
                self.targets_and_models.append((target.model, module.model))

        self.actor_opt = torch.optim.Adam(
            actor_params,
            lr=self.config.actor_lr,
            weight_decay=self.actor_weight_decay)

        self.actor_params = actor_params

        # Critic setup
        critic_params = []
        self.critics = []
        for module in list(self.module_dict.values()):
            name = module.module_name
            if name.startswith(self.critic_name) and isinstance(module, PytorchModel):
                self.critics.append(module)
                critic_params += list(module.model.parameters())
                target = module.copy(name + '_target')
                target.model.load_state_dict(module.model.state_dict())
                # Freeze target networks with respect to optimizers (only update via polyak averaging)
                for p in target.model.parameters():
                    p.requires_grad = False
                self.agent.set_module(name + '_target', target)
                self.targets_and_models.append((target.model, module.model))

        self.critic_opt = torch.optim.Adam(
            critic_params,
            lr=self.config.critic_lr,
            weight_decay=self.critic_weight_decay)

        self.critic_params = critic_params

        self.actor_algo = getattr(self, self.actor_name)
        self.critic_algo = getattr(self, self.critic_name)
        self.actor_algo_target = getattr(self, self.actor_name + '_target')
        self.critic_algo_target = getattr(self, self.critic_name + '_target')
        self.replay_buffer = getattr(self, self.replay_buffer_name)

        self.action_scale = self.env.max_action

    def _optimize(self):
        if len(self.replay_buffer) > self.config.warm_up:
            self.step += 1
            if self.step % self.optimize_every == 0:
                states, actions, rewards, next_states, gammas = self.replay_buffer.sample(
                    self.config.batch_size, append_context=self.is_explore)

                self.optimize_from_batch(states, actions, rewards, next_states, gammas)

                if self.config.opt_steps % self.target_network_update_freq == 0:
                    for target_model, model in self.targets_and_models:
                        soft_update(target_model, model, self.target_network_update_frac)


class DDPG2(OffPolicyActorCritic2):

    def optimize_from_batch(self, states, actions, rewards, next_states, gammas):

        with torch.no_grad():
            q_next = self.critic_algo_target(next_states, self.actor_algo_target(next_states))
            target = (rewards + gammas * q_next)
            if self.clip_target_range:
                target = torch.clamp(target, *self.clip_target_range)

        if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
            self.logger.add_histogram(f'Optimize/{self.module_name}/Target_q', target)

        q = self.critic_algo(states, actions)
        critic_loss = F.mse_loss(q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()

        # Grad clipping
        if self.config.grad_norm_clipping > 0.:
            for p in self.critic_params:
                clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
                if clip_coef < 1:
                    p.grad.detach().mul_(clip_coef)
        if self.config.grad_value_clipping > 0.:
            torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

        self.critic_opt.step()

        for p in self.critic_params:
            p.requires_grad = False

        a = self.actor_algo(states)
        if self.config.get('policy_opt_noise'):
            noise = torch.randn_like(a) * (self.config.policy_opt_noise * self.action_scale)
            a = (a + noise).clamp(-self.action_scale, self.action_scale)

        actor_loss = -self.critic_algo(states, a)[:, -1].mean()
        if self.config.action_l2_regularization:
            actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))

        self.actor_opt.zero_grad()
        actor_loss.backward()

        # Grad clipping
        if self.config.grad_norm_clipping > 0.:
            for p in self.actor_params:
                clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
                if clip_coef < 1:
                    p.grad.detach().mul_(clip_coef)
        if self.config.grad_value_clipping > 0.:
            torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)

        self.actor_opt.step()

        for p in self.critic_params:
            p.requires_grad = True

        if hasattr(self, 'logger'):
            self.logger.add_scalar(f'Optimize/{self.module_name}/critic_loss', critic_loss.item())
            self.logger.add_scalar(f'Optimize/{self.module_name}/actor_loss', actor_loss.item())


class SAC2(OffPolicyActorCritic2):

    def _setup(self):
        super()._setup()
        self.critic2_algo = getattr(self, self.critic_name+'2')
        self.critic2_algo_target = getattr(self, self.critic_name + '2_target')

    def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
        config = self.config

        with torch.no_grad():
            # Target actions come from *current* policy
            a_next, logp_next = self.actor_algo(next_states)
            q1 = self.critic_algo_target(next_states, a_next)
            q2 = self.critic2_algo_target(next_states, a_next)
            target = config.reward_scale * rewards + gammas * (torch.min(q1, q2) - config.entropy_coef * logp_next)
            if self.clip_target_range:
                target = torch.clamp(target, *self.clip_target_range)

        if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
            self.logger.add_histogram(f'Optimize/{self.module_name}/Target_q', target)

        q1, q2 = self.critic_algo(states, actions), self.critic2_algo(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()

        # Grad clipping
        if self.config.grad_norm_clipping > 0.:
            torch.nn.utils.clip_grad_norm_(self.critic_params, self.config.grad_norm_clipping)
        if self.config.grad_value_clipping > 0.:
            torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

        self.critic_opt.step()

        for p in self.critic_params:
            p.requires_grad = False

        a, logp = self.actor_algo(states)
        q = torch.min(self.critic_algo(states, a), self.critic2_algo(states, a))

        actor_loss = (config.entropy_coef * logp - q).mean()

        if self.config.action_l2_regularization:
            actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))

        self.actor_opt.zero_grad()
        actor_loss.backward()

        # Grad clipping
        if self.config.grad_norm_clipping > 0.:
            torch.nn.utils.clip_grad_norm_(self.actor_params, self.config.grad_norm_clipping)
        if self.config.grad_value_clipping > 0.:
            torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)

        self.actor_opt.step()

        for p in self.critic_params:
            p.requires_grad = True

        if hasattr(self, 'logger'):
            self.logger.add_scalar(f'Optimize/{self.module_name}/q1', q1.mean().item())
            self.logger.add_scalar(f'Optimize/{self.module_name}/q2', q2.mean().item())
            self.logger.add_scalar(f'Optimize/{self.module_name}/logp', logp.mean().item())
            self.logger.add_scalar(f'Optimize/{self.module_name}/critic_loss', critic_loss.item())
            self.logger.add_scalar(f'Optimize/{self.module_name}/actor_loss', actor_loss.item())
