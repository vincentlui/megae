import mrl
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn

from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.utils.misc import softmax, AttrDict
from mrl.utils.networks import layer_init


class Empowerment(mrl.Module):
    """
      Empowerment intrinsic module
    """

    def __init__(self, policy, optimize_every=1):
        super().__init__('empowerment',
                         required_agent_modules=['env', 'replay_buffer', 'actor', 'critic'],
                         locals=locals())
        self.behavior_policy = policy
        self.optimize_every = optimize_every
        self.step = 0

    # def _setup(self):
    #     # self.inverse_dyn_params = self.inverse_dyn.model.parameters()
    #     # self.inverse_dyn_opt = torch.optim.Adam(
    #     #     self.inverse_dyn_params,
    #     #     lr=self.config.actor_lr, #inverse_dyn_lr,
    #     #     weight_decay=self.config.actor_weight_decay) #inverse_dyn_weight_decay)
    #     self.step = 0

    def _optimize(self):
        raise NotImplementedError  # SUBCLASS THIS!

    def calc_empowerment(self, actions, states, next_states):
        raise NotImplementedError  # SUBCLASS THIS!
        # log q(a|s,s') - log pi
        # input = np.concatenate([actions, states, next_states], axis=-1)
        # input = self.torch(input)
        # logq = self.inverse_dyn(input)
        # _, logp = self.behavior_policy(self.torch(states))
        # return self.numpy(logq - logp)


class InverseDynamicsNetwork(nn.Module):
    def __init__(self, body: nn.Module):
        super().__init__()
        self.body = body
        self.fc = layer_init(nn.Linear(self.body.feature_dim, 1))

    def forward(self, x, actions):
        # mu_and_log_std = self.fc(self.body(x))
        # mu, log_std = torch.chunk(mu_and_log_std, 2, -1)
        #
        # action = mu
        # logp_action = None
        # if self.training:
        #     log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        #     std = torch.exp(log_std)
        #     action_distribution = Normal(mu, std)
        #     action = action_distribution.rsample()
        #     logp_action = action_distribution.log_prob(action).sum(axis=-1, keepdims=True)
        #     logp_action -= (2 * (self.log2 - action - F.softplus(-2 * action))).sum(axis=1, keepdims=True)
        #
        # action = torch.tanh(action)
        # return self.max_action * action, logp_action
        return torch.sigmoid(self.fc(self.body(x)))

class JSMI(Empowerment):
    def __init__(self, policy, optimize_every=1):
        super().__init__(policy, optimize_every)
        self.log4 = np.log(4.)

    def _setup(self):
        self.T_params = self.T.model.parameters()
        self.T_opt = torch.optim.Adam(
            self.T_params,
            lr=self.config.emp_lr,
            weight_decay=self.config.actor_weight_decay)

    def _optimize(self):
        # sample
        if len(self.replay_buffer) > self.config.warm_up:
            self.step += 1
            if self.step % self.optimize_every == 0:
                states, actions, rewards, next_states, dones, contexts, next_contexts, reward_expls, _, previous_ags, ags, goals, _ = \
                    self.replay_buffer.buffer.sample(self.config.batch_size)
                states = np.concatenate([states, contexts], axis=-1)
                if hasattr(self, 'state_normalizer_expl'):
                    states = self.state_normalizer_expl(states, update=False).astype(np.float32)
                states = self.torch(states)
                actions = self.torch(actions)
                ags = self.torch(ags)
                input = torch.cat([actions, states, ags], dim=-1)
                # if hasattr(self, 'state_normalizer_empowerment'):
                #     states = self.state_normalizer_empowerment(input, update=False).astype(np.float32)
                #     next_states = self.state_normalizer_expl(
                #         next_states, update=False).astype(np.float32)
                with torch.no_grad():
                    a_policy, _ = self.behavior_policy(states)
                input2 = torch.cat([a_policy, states, ags], dim=-1)
                T1 = self.T(input)
                T2 = self.T(input2)
                if self.config.clip_empowerment:
                    T1 = torch.clip(T1, -self.config.clip_empowerment, self.config.clip_empowerment)
                    T2 = torch.clip(T2, -self.config.clip_empowerment, self.config.clip_empowerment)
                loss = torch.mean(F.softplus(-T1) + F.softplus(T2) - self.log4)

                if hasattr(self, 'logger'):
                    self.logger.add_scalar('Optimize/JSMI_loss', loss.item())

                self.T_opt.zero_grad()
                loss.backward()

                # Grad clipping
                if self.config.grad_norm_clipping > 0.:
                    for p in self.T_params:
                        clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
                        if clip_coef < 1:
                            p.grad.detach().mul_(clip_coef)
                if self.config.grad_value_clipping > 0.:
                    torch.nn.utils.clip_grad_value_(self.T_params, self.config.grad_value_clipping)

                self.T_opt.step()

    def calc_empowerment(self, actions, states, next_states):
        if not isinstance(actions, torch.Tensor):
            actions = self.torch(actions)
            states = self.torch(states)
            next_states = self.torch(next_states)
        input = torch.cat([actions, states, next_states], dim=-1)
        empowerment = self.numpy(self.T(input))
        if self.config.clip_empowerment:
            empowerment = np.clip(empowerment, -self.config.clip_empowerment, self.config.clip_empowerment)
        return empowerment

    def save(self, save_folder: str):
        path = os.path.join(save_folder, self.module_name + '.pt')
        torch.save({
            'T_opt_state_dict': self.T_opt.state_dict(),
        }, path)

    def load(self, save_folder: str):
        path = os.path.join(save_folder, self.module_name + '.pt')
        checkpoint = torch.load(path,
                                map_location=torch.device(self.config.device))  # , map_location=self.config.device)
        self.T_opt.load_state_dict(checkpoint['T_opt_state_dict'])



class JSMIT(nn.Module):
    def __init__(self, body: nn.Module):
        super().__init__()
        self.body = body
        self.fc = layer_init(nn.Linear(self.body.feature_dim, 1))

    def forward(self, x):
        return self.fc(self.body(x))
