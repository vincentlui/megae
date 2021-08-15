"""
Curiosity modules for unsupervised exploration
"""

import mrl
import numpy as np
import torch
from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.utils.misc import softmax, AttrDict
from sklearn.neighbors import KernelDensity
from collections import deque
from numpy.random import multivariate_normal, uniform


def generate_overshooting_goals(num_proposals, step_amount, direct_overshoots, base_goal):
    base_proposals = np.array([base_goal, base_goal + step_amount])
    if direct_overshoots:
        return base_proposals
    additional_proposals = base_goal[None] + np.random.uniform(
        -1.5, 1.5, (num_proposals - 2, step_amount.shape[0])) * step_amount[None]
    return np.concatenate((base_proposals, additional_proposals), 0)


class MegaeCuriosity(mrl.Module):
    """
      For goal agents only. This module assumes the replay buffer maintains an achieved goal buffer;
      To decide on goals to pursue during exploration, the module samples goals from the achieved goal
      buffer, and chooses the highest scoring (see below) viable (per q-function) goal.
    """

    def __init__(self, num_sampled_ags=500, max_steps=50, keep_dg_percent=-1e-1, randomize=False, use_qcutoff=True,
                 exploration_percent=0.8, num_context=10, context_var=0.1, context_dist='normal',
                 initial_explore_percent=0.05, intermediate_goal_exploration=False):
        super().__init__('ag_curiosity',
                         required_agent_modules=['env', 'replay_buffer', 'actor', 'critic'],
                         locals=locals())
        self.num_sampled_ags = num_sampled_ags
        self.max_steps = max_steps  # TODO: have this be learned from past trajectories?
        self.keep_dg_percent = keep_dg_percent
        self.randomize = randomize
        self.use_qcutoff = use_qcutoff
        self.explortation_start_steps = int(max_steps * exploration_percent)
        self.num_context = num_context
        self.context_var = context_var
        self.context_dist = context_dist
        self.initial_explore_percent = initial_explore_percent
        self.intermediate_goal_exploration = intermediate_goal_exploration

    def _setup(self):
        assert isinstance(self.replay_buffer, OnlineHERBuffer)
        assert self.env.goal_env

        self.n_envs = self.env.num_envs
        self.current_goals = None
        self.intermediate_goals = None
        self.replaced_goal = np.zeros((self.env.num_envs,))

        # setup cutoff
        if self.config.gamma < 1.:
            r = min(self.config.gamma, 0.99)
            self.min_min_cutoff = -(1 - r ** (self.max_steps * 0.8)) / (1 - r)
        else:
            self.min_min_cutoff = -self.max_steps * 0.8
        self.min_cutoff = max(self.config.initial_cutoff, self.min_min_cutoff)
        self.cutoff = self.min_cutoff

        # go explore + success accounting
        self.go_explore = np.zeros((self.n_envs, 1), dtype=np.float32)
        self.is_success = np.zeros((self.n_envs, 1), dtype=np.float32)
        self.successes_deque = deque(maxlen=10)  # for dynamic cutoff
        self.successes = []
        self.ag_from = np.zeros((self.n_envs, self.env.goal_dim), dtype=np.float32)
        self.intermediate_reached = np.full((self.n_envs, ), False)

        # context
        self.context_states = self._generate_context_states(self.num_context, self.context_var)

        # number of environment steps
        self.num_steps = np.zeros((self.n_envs, 1), dtype=np.float32)
        self.is_explore = np.zeros((self.n_envs,1), dtype=np.bool)

    def _manage_resets_and_success_behaviors(self, experience, close):
        """ Manage (1) end of trajectory, (2) early resets, (3) go explore and overshot goals """
        reset_idxs, overshooting_idxs, overshooting_proposals = [], [], []

        for i, over in enumerate(experience.trajectory_over):
            if over:  # if over update it
                self.current_goals[i] = experience.reset_state['desired_goal'][i]
                self.replaced_goal[i] = 0.
                if np.random.random() < (self.go_explore[i] * self.config.go_reset_percent):
                    reset_idxs.append(i)

            if not over and close[i]:  # if not over and success, modify go_explore; maybe overshoot goal?
                if self.intermediate_goal_exploration and not self.intermediate_reached[i]:
                    self.intermediate_reached[i] = True
                else:
                    self.is_success[i] += 1.
                    self.go_explore[i] += 1.

                    if not self.is_explore[i] and ( np.random.random() < self.is_success[i] * 5./self.max_steps): #self.num_steps[i] >= self.explortation_start_steps or
                        self.is_explore[i] = True
                        self.ag_from[i] = experience.next_state['achieved_goal'][i]

        return reset_idxs, overshooting_idxs, np.array(overshooting_proposals)

    def _overshoot_goals(self, experience, overshooting_idxs, overshooting_proposals):
        # score the proposals
        num_proposals = overshooting_proposals.shape[1]
        num_idxs = len(overshooting_idxs)
        states = np.tile(experience.reset_state['observation'][overshooting_idxs, None, :], (1, num_proposals, 1))
        states = np.concatenate((states, overshooting_proposals), -1).reshape(num_proposals * num_idxs, -1)

        bad_q_idxs, q_values = [], None
        if self.use_qcutoff:
            q_values = self.compute_q(states)
            q_values = q_values.reshape(num_idxs, num_proposals)
            bad_q_idxs = q_values < self.cutoff
        goal_values = self.score_goals(overshooting_proposals, AttrDict(q_values=q_values, states=states))

        if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
            dg_scores = self.dg_kde.evaluate_log_density(overshooting_proposals.reshape(num_proposals * num_idxs, -1))
            dg_scores = dg_scores.reshape(num_idxs, num_proposals)
            goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier

        goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

        chosen_idx = np.argmin(goal_values, axis=1)
        chosen_idx = np.eye(num_proposals)[chosen_idx]  # shape(sampled_ags) = n_envs x num_proposals
        chosen_ags = np.sum(overshooting_proposals * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats

        for idx, goal in zip(overshooting_idxs, chosen_ags):
            self.current_goals[idx] = goal
            self.replaced_goal[idx] = 1.

    def _process_experience(self, experience):
        """Curiosity module updates the desired goal depending on experience.trajectory_over"""
        ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag
        is_explore_buffer = self.replay_buffer.buffer.BUFF.buffer_is_explore
        ig_buffer = self.replay_buffer.buffer.BUFF.buffer_ig

        self.num_steps += 1.

        if self.current_goals is None:
            self.current_goals = experience.reset_state['desired_goal']

        goals = self.current_goals.copy()
        if self.intermediate_goal_exploration:
            if self.intermediate_goals is None:
                self.intermediate_goals = self.current_goals
            goals[~self.intermediate_reached] = self.intermediate_goals[~self.intermediate_reached]

        computed_reward = self.env.compute_reward(experience.next_state['achieved_goal'], goals,
                                                  {'s': experience.state['observation'],
                                                   'ns': experience.next_state['observation']})
        close = computed_reward > -0.5

        # First, manage the episode resets & any special behavior that occurs on goal achievement, like go explore / resets / overshooting
        reset_idxs, overshooting_idxs, overshooting_proposals = self._manage_resets_and_success_behaviors(experience,
                                                                                                          close)

        if reset_idxs:
            self.train.reset_next(reset_idxs)

        if overshooting_idxs and len(ag_buffer):
            self._overshoot_goals(experience, overshooting_idxs, overshooting_proposals)

        # Now consider replacing the current goals with something else:
        if np.any(experience.trajectory_over) and len(ag_buffer):
            # sample some achieved goals
            sample_idxs = np.random.randint(len(ag_buffer), size=self.num_sampled_ags * self.n_envs)
            sampled_ags = ag_buffer.get_batch(sample_idxs).reshape(self.n_envs, self.num_sampled_ags, -1)
            sampled_is_explore = is_explore_buffer.get_batch(sample_idxs).reshape(self.n_envs, self.num_sampled_ags, -1)
            sampled_ig = ig_buffer.get_batch(sample_idxs).reshape(self.n_envs, self.num_sampled_ags, -1)

            # compute the q-values of both the sampled achieved goals and the current goals
            states = np.tile(experience.reset_state['observation'][:, None, :], (1, self.num_sampled_ags, 1))
            states = np.concatenate((states, sampled_ags), -1).reshape(self.num_sampled_ags * self.n_envs, -1)
            states_curr = np.concatenate((experience.reset_state['observation'], self.current_goals), -1)
            states_cat = np.concatenate((states, states_curr), 0)
            if hasattr(self, 'state_normalizer'):
                states_cat = self.state_normalizer(states_cat, update=False).astype(np.float32)

            bad_q_idxs, q_values = [], None
            if self.use_qcutoff:
                q_values = self.compute_q(states_cat)
                q_values, curr_q = np.split(q_values, [self.num_sampled_ags * self.n_envs])
                q_values = q_values.reshape(self.n_envs, self.num_sampled_ags)

                # Set cutoff dynamically by using intrinsic_success_percent
                if len(self.successes_deque) == 10:
                    self.min_cutoff = max(self.min_min_cutoff, min(np.min(q_values), self.min_cutoff))
                    intrinsic_success_percent = np.mean(self.successes_deque)
                    if intrinsic_success_percent >= self.config.cutoff_success_threshold[1]:
                        self.cutoff = max(self.min_cutoff, self.cutoff - 1.)
                        self.successes_deque.clear()
                    elif intrinsic_success_percent <= self.config.cutoff_success_threshold[0]:
                        self.cutoff = max(min(self.config.initial_cutoff, self.cutoff + 1.), self.min_min_cutoff)
                        self.successes_deque.clear()

                # zero out the "bad" values. This practically eliminates them as candidates if any goals are viable.
                bad_q_idxs = q_values < self.cutoff
                q_values[bad_q_idxs] *= -1
                min_q_values = np.min(q_values, axis=1, keepdims=True)  # num_envs x1
                q_values[bad_q_idxs] *= -1

            # score the goals -- lower is better
            goal_values = self.score_goals(sampled_ags, AttrDict(q_values=q_values, states=states))

            if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
                dg_scores = self.dg_kde.evaluate_log_density(
                    sampled_ags.reshape(self.n_envs * self.num_sampled_ags, -1))
                dg_scores = dg_scores.reshape(self.n_envs, self.num_sampled_ags)
                goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier

            if q_values is not None:
                goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

            if self.randomize:  # sample proportional to the absolute score
                abs_goal_values = np.abs(goal_values)
                normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=1, keepdims=True)
                chosen_idx = (normalized_values.cumsum(1) > np.random.rand(normalized_values.shape[0])[:, None]).argmax(
                    1)
            else:  # take minimum
                chosen_idx = np.argmin(goal_values, axis=1)

            chosen_idx = np.eye(self.num_sampled_ags)[chosen_idx]  # shape(sampled_ags) = n_envs x num_sampled_ags
            if q_values is not None:
                chosen_q_val = (chosen_idx * q_values).sum(axis=1, keepdims=True)
            chosen_ags = np.sum(sampled_ags * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats
            chosen_is_explore = np.sum(sampled_is_explore * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats
            chosen_igs = np.sum(sampled_ig * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats

            # replace goal always when first_visit_succ (relying on the dg_score_multiplier to dg focus), otherwise
            # we are going to transition into the dgs using the ag_kde_tophat
            if hasattr(self, 'curiosity_alpha'):
                if self.use_qcutoff:
                    replace_goal = np.logical_or((np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha),
                                                 curr_q < self.cutoff).astype(np.float32)
                else:
                    replace_goal = (np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha).astype(np.float32)

            else:
                replace_goal = np.ones((self.n_envs, 1), dtype=np.float32)

            # sometimes keep the desired goal anyways
            replace_goal *= (np.random.uniform(size=[self.n_envs, 1]) > self.keep_dg_percent).astype(np.float32)

            new_goals = replace_goal * chosen_ags + (1 - replace_goal) * self.current_goals
            new_igs = None
            if self.intermediate_goal_exploration:
                new_igs = replace_goal * chosen_igs + (1 - replace_goal) * self.intermediate_goals

            if hasattr(self, 'logger') and len(self.successes) > 50:
                if q_values is not None:
                    self.logger.add_histogram('Explore/Goal_q',
                                              replace_goal * chosen_q_val + (1 - replace_goal) * curr_q)
                self.logger.add_scalar('Explore/Intrinsic_success_percent', np.mean(self.successes))
                self.logger.add_scalar('Explore/Cutoff', self.cutoff)
                self.successes = []

            replace_goal = replace_goal.reshape(-1)

            for i in range(self.n_envs):
                if experience.trajectory_over[i]:
                    self.successes.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
                    self.successes_deque.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
                    self.current_goals[i] = new_goals[i]
                    if replace_goal[i]:
                        self.replaced_goal[i] = 1.
                    self.go_explore[i] = 0.
                    self.is_success[i] = 0.
                    self.is_explore[i] = np.random.uniform() < self.initial_explore_percent
                    self.num_steps[i] = 0.
                    self.intermediate_reached[i] = False
                    self.intermediate_goals = new_igs
                    self.ag_from = np.zeros_like(self.ag_from)

    def _generate_context_states(self, num_context, context_var):
        goal_dim = self.env.goal_dim
        if self.context_dist == 'normal':
            mean = np.zeros(goal_dim)
            cov = np.identity(goal_dim) * context_var
            return multivariate_normal(mean, cov, num_context)

        return uniform(-context_var, context_var, (num_context, goal_dim))

    def compute_q(self, numpy_states):
        states = self.torch(numpy_states)
        max_actions = self.actor(states)
        if isinstance(max_actions, tuple):
            max_actions = max_actions[0]
        return self.numpy(self.critic(states, max_actions))

    def relabel_state(self, state):
        """Should be called by the policy module to relabel states with intrinsic goals"""
        if self.current_goals is None:
            return state

        desired_goal = self.current_goals.copy()
        if self.intermediate_goal_exploration:
            desired_goal[~self.intermediate_reached] = self.intermediate_goals[~self.intermediate_reached]

        return {
            'observation': state['observation'],
            'achieved_goal': state['achieved_goal'],
            'desired_goal': desired_goal,
            'behavioral_goal': self.current_goals,
        }

    def score_goals(self, sampled_ags, info):
        """ Lower is better """
        raise NotImplementedError  # SUBCLASS THIS!

    def score_states(self, states):
        """ Lower is better """
        raise NotImplementedError  # SUBCLASS THIS!

    def save(self, save_folder):
        self._save_props(['cutoff', 'min_cutoff'], save_folder)  # can restart keeping track of successes / go explore

    def load(self, save_folder):
        self._load_props(['cutoff', 'min_cutoff'], save_folder)

class DensityMegaeCuriosity(MegaeCuriosity):
  """
  Scores goals by their densities (lower is better), using KDE to estimate

  Note on bandwidth: it seems bandwith = 0.1 works pretty well with normalized samples (which is
  why we normalize the ags).
  """
  def __init__(self, density_module='ag_kde', interest_module='ag_interest', alpha=-1.0, **kwargs):
    super().__init__(**kwargs)
    self.alpha = alpha
    self.density_module = density_module
    self.interest_module = interest_module

  def _setup(self):
    assert hasattr(self, self.density_module)
    super()._setup()

  def relabel_state(self, state):
    """Should be called by the policy module to relabel states with intrinsic goals"""
    if self.current_goals is None:
        return state

    desired_goal = self.current_goals

    return {
        'observation': state['observation'],
        'achieved_goal': state['achieved_goal'],
        'desired_goal': self.current_goals
    }

  def score_goals(self, sampled_ags, info):
    """ Lower is better """
    density_module = getattr(self, self.density_module)
    if not density_module.ready:
      density_module._optimize(force=True)
    interest_module = None
    if hasattr(self, self.interest_module):
      interest_module = getattr(self, self.interest_module)
      if not interest_module.ready:
        interest_module = None

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    # score the sampled_ags to get log densities, and exponentiate to get densities
    flattened_sampled_ags = sampled_ags.reshape(num_envs * num_sampled_ags, -1)
    sampled_ag_scores = density_module.evaluate_log_density(flattened_sampled_ags)
    if interest_module:
      # Interest is ~(det(feature_transform)), so we subtract it  in order to add ~(det(inverse feature_transform)) for COV.
      sampled_ag_scores -= interest_module.evaluate_log_interest(flattened_sampled_ags)  # add in log interest
    sampled_ag_scores = sampled_ag_scores.reshape(num_envs, num_sampled_ags)  # these are log densities

    if self.config.get('clip_density'):
        sampled_ag_scores = np.clip(sampled_ag_scores, -self.config.clip_density, self.config.clip_density)

    # Take softmax of the alpha * log density.
    # If alpha = -1, this gives us normalized inverse densities (higher is rarer)
    # If alpha < -1, this skews the density to give us low density samples
    normalized_inverse_densities = softmax(sampled_ag_scores * self.alpha, axis=-1)
    normalized_inverse_densities *= -1.  # make negative / reverse order so that lower is better.

    return normalized_inverse_densities

  def score_states(self, states):
      ag = states['achieved_goal']
      density_module = getattr(self, self.density_module)
      if not density_module.ready:
          # density_module._optimize(force=True)
        return np.zeros(ag.shape[0])
      states_score = -1 * density_module.evaluate_log_density(ag.astype(np.float32))
      if self.config.get('clip_density'):
        states_score = np.clip(states_score, -self.config.clip_density, self.config.clip_density)

      return states_score

  def get_context(self, states):
      ag = states['achieved_goal']
      num_envs = ag.shape[0]

      density_module = getattr(self, self.density_module)
      if not density_module.ready:
          # density_module._optimize(force=True)
        return np.ones((num_envs, self.num_context)) / self.num_context

      ag_tile = np.tile(ag, (self.num_context, )).reshape(num_envs, self.num_context, -1)
      context_states = ag_tile + self.context_states
      flattened_context_states = context_states.reshape(num_envs * self.num_context, -1).astype(np.float32)
      density_context_states = softmax(density_module.evaluate_log_density(flattened_context_states)\
          .reshape(num_envs, self.num_context), axis=-1)
      density_context_states_normalized = density_context_states #/ np.linalg.norm(density_context_states)
      return density_context_states_normalized


class DensityAndExplorationMegaeCuriosity(MegaeCuriosity):
  """
  Scores goals by their densities (lower is better), using KDE to estimate

  Note on bandwidth: it seems bandwith = 0.1 works pretty well with normalized samples (which is
  why we normalize the ags).
  """
  def __init__(self, density_module='ag_kde', interest_module='ag_interest', alpha=-1.0, density_percent=0.5, **kwargs):
    super().__init__(**kwargs)
    self.alpha = alpha
    self.density_module = density_module
    self.interest_module = interest_module
    self.density_percent = density_percent

  def _setup(self):
    assert hasattr(self, self.density_module)
    super()._setup()

  def _process_experience(self, experience):
      """Curiosity module updates the desired goal depending on experience.trajectory_over"""
      ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag
      self.num_steps += 1.

      if self.current_goals is None:
          self.current_goals = experience.reset_state['desired_goal']

      computed_reward = self.env.compute_reward(experience.next_state['achieved_goal'], self.current_goals,
                                                {'s': experience.state['observation'],
                                                 'ns': experience.next_state['observation']})
      close = computed_reward > -0.5

      # First, manage the episode resets & any special behavior that occurs on goal achievement, like go explore / resets / overshooting
      reset_idxs, overshooting_idxs, overshooting_proposals = self._manage_resets_and_success_behaviors(experience,
                                                                                                        close)

      if reset_idxs:
          self.train.reset_next(reset_idxs)

      if overshooting_idxs and len(ag_buffer):
          self._overshoot_goals(experience, overshooting_idxs, overshooting_proposals)

      # Now consider replacing the current goals with something else:
      if np.any(experience.trajectory_over) and len(self.replay_buffer):
          # how many density samples and exploration samples
          num_density, num_exploration = np.random.multinomial(self.n_envs, [self.density_percent, 1. - self.density_percent])

          scores = np.empty((self.n_envs, self.num_sampled_ags))
          sampled_ags = np.empty((self.n_envs, self.num_sampled_ags, self.env.goal_dim))
          # sample some achieved goals
          if num_exploration:
              s, _, _, _, _, _, _, \
              _, _, previous_ags, ags, _, _, _ = self.replay_buffer.buffer.sample(self.num_sampled_ags * num_exploration)
              s_explore = np.concatenate([s, self.get_context({'achieved_goal': ags})], axis=-1)
              if hasattr(self, 'state_normalizer_expl'):
                  s_explore = self.state_normalizer_expl(s_explore, update=False).astype(np.float32)
              score_expl = self.score_goals_expl(s_explore.reshape(num_exploration, self.num_sampled_ags, -1), None)
              scores[np.arange(num_exploration)] = score_expl
              sampled_ags[np.arange(num_exploration)] = previous_ags.reshape(num_exploration, self.num_sampled_ags, -1)

          if num_density:
              sample_idxs = np.random.randint(len(ag_buffer), size=self.num_sampled_ags * num_density)
              sampled_ags_density = ag_buffer.get_batch(sample_idxs)
              sampled_ags_density = sampled_ags_density.reshape(num_density, self.num_sampled_ags, -1)
              score_density = self.score_goals_density(sampled_ags_density, None)
              scores[np.arange(num_exploration, self.n_envs)] = score_density
              sampled_ags[np.arange(num_exploration, self.n_envs)] = sampled_ags_density

          # compute the q-values of both the sampled achieved goals and the current goals
          states = np.tile(experience.reset_state['observation'][:, None, :], (1, self.num_sampled_ags, 1))
          states = np.concatenate((states, sampled_ags), -1).reshape(self.num_sampled_ags * self.n_envs, -1)
          states_curr = np.concatenate((experience.reset_state['observation'], self.current_goals), -1)
          states_cat = np.concatenate((states, states_curr), 0)
          if hasattr(self, 'state_normalizer'):
              states_cat = self.state_normalizer(states_cat, update=False).astype(np.float32)

          bad_q_idxs, q_values = [], None
          if self.use_qcutoff:
              q_values = self.compute_q(states_cat)
              q_values, curr_q = np.split(q_values, [self.num_sampled_ags * self.n_envs])
              q_values = q_values.reshape(self.n_envs, self.num_sampled_ags)

              # Set cutoff dynamically by using intrinsic_success_percent
              if len(self.successes_deque) == 10:
                  self.min_cutoff = max(self.min_min_cutoff, min(np.min(q_values), self.min_cutoff))
                  intrinsic_success_percent = np.mean(self.successes_deque)
                  if intrinsic_success_percent >= self.config.cutoff_success_threshold[1]:
                      self.cutoff = max(self.min_cutoff, self.cutoff - 1.)
                      self.successes_deque.clear()
                  elif intrinsic_success_percent <= self.config.cutoff_success_threshold[0]:
                      self.cutoff = max(min(self.config.initial_cutoff, self.cutoff + 1.), self.min_min_cutoff)
                      self.successes_deque.clear()

              # zero out the "bad" values. This practically eliminates them as candidates if any goals are viable.
              bad_q_idxs = q_values < self.cutoff
              q_values[bad_q_idxs] *= -1
              min_q_values = np.min(q_values, axis=1, keepdims=True)  # num_envs x1
              q_values[bad_q_idxs] *= -1

          # score the goals -- lower is better
          goal_values = scores

          if self.config.dg_score_multiplier > 1. and self.dg_kde.ready:
              dg_scores = self.dg_kde.evaluate_log_density(
                  sampled_ags.reshape(self.n_envs * self.num_sampled_ags, -1))
              dg_scores = dg_scores.reshape(self.n_envs, self.num_sampled_ags)
              goal_values[dg_scores > -np.inf] *= self.config.dg_score_multiplier

          if q_values is not None:
              goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

          if self.randomize:  # sample proportional to the absolute score
              abs_goal_values = np.abs(goal_values)
              normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=1, keepdims=True)
              chosen_idx = (normalized_values.cumsum(1) > np.random.rand(normalized_values.shape[0])[:, None]).argmax(
                  1)
          else:  # take minimum
              chosen_idx = np.argmin(goal_values, axis=1)

          chosen_idx = np.eye(self.num_sampled_ags)[chosen_idx]  # shape(sampled_ags) = n_envs x num_sampled_ags
          if q_values is not None:
              chosen_q_val = (chosen_idx * q_values).sum(axis=1, keepdims=True)
          chosen_ags = np.sum(sampled_ags * chosen_idx[:, :, None], axis=1)  # n_envs x goal_feats

          # replace goal always when first_visit_succ (relying on the dg_score_multiplier to dg focus), otherwise
          # we are going to transition into the dgs using the ag_kde_tophat
          if hasattr(self, 'curiosity_alpha'):
              if self.use_qcutoff:
                  replace_goal = np.logical_or((np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha),
                                               curr_q < self.cutoff).astype(np.float32)
              else:
                  replace_goal = (np.random.random((self.n_envs, 1)) > self.curiosity_alpha.alpha).astype(np.float32)

          else:
              replace_goal = np.ones((self.n_envs, 1), dtype=np.float32)

          # sometimes keep the desired goal anyways
          replace_goal *= (np.random.uniform(size=[self.n_envs, 1]) > self.keep_dg_percent).astype(np.float32)

          new_goals = replace_goal * chosen_ags + (1 - replace_goal) * self.current_goals

          if hasattr(self, 'logger') and len(self.successes) > 50:
              if q_values is not None:
                  self.logger.add_histogram('Explore/Goal_q',
                                            replace_goal * chosen_q_val + (1 - replace_goal) * curr_q)
              self.logger.add_scalar('Explore/Intrinsic_success_percent', np.mean(self.successes))
              self.logger.add_scalar('Explore/Cutoff', self.cutoff)
              self.successes = []

          replace_goal = replace_goal.reshape(-1)

          for i in range(self.n_envs):
              if experience.trajectory_over[i]:
                  self.successes.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
                  self.successes_deque.append(float(self.is_success[i, 0] >= 1.))  # compromise due to exploration
                  self.current_goals[i] = new_goals[i]
                  if replace_goal[i]:
                      self.replaced_goal[i] = 1.
                  self.go_explore[i] = 0.
                  self.is_success[i] = 0.
                  self.is_explore[i] = np.random.uniform() < self.initial_explore_percent
                  self.num_steps[i] = 0.

  def score_goals_density(self, sampled_ags, info):
    """ Lower is better """
    density_module = getattr(self, self.density_module)
    if not density_module.ready:
      density_module._optimize(force=True)
    interest_module = None
    if hasattr(self, self.interest_module):
      interest_module = getattr(self, self.interest_module)
      if not interest_module.ready:
        interest_module = None

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_ags.shape[:2]

    # score the sampled_ags to get log densities, and exponentiate to get densities
    flattened_sampled_ags = sampled_ags.reshape(num_envs * num_sampled_ags, -1)
    sampled_ag_scores = density_module.evaluate_log_density(flattened_sampled_ags)
    if interest_module:
      # Interest is ~(det(feature_transform)), so we subtract it  in order to add ~(det(inverse feature_transform)) for COV.
      sampled_ag_scores -= interest_module.evaluate_log_interest(flattened_sampled_ags)  # add in log interest
    sampled_ag_scores = sampled_ag_scores.reshape(num_envs, num_sampled_ags)  # these are log densities

    # Take softmax of the alpha * log density.
    # If alpha = -1, this gives us normalized inverse densities (higher is rarer)
    # If alpha < -1, this skews the density to give us low density samples
    normalized_inverse_densities = softmax(sampled_ag_scores * self.alpha, axis=-1)
    normalized_inverse_densities *= -1.  # make negative / reverse order so that lower is better.

    return normalized_inverse_densities

  def score_goals_expl(self, sampled_states, info):
    """ Lower is better """
    density_module = getattr(self, self.density_module)
    if not density_module.ready:
      density_module._optimize(force=True)
    interest_module = None
    if hasattr(self, self.interest_module):
      interest_module = getattr(self, self.interest_module)
      if not interest_module.ready:
        interest_module = None

    # sampled_ags is np.array of shape NUM_ENVS x NUM_SAMPLED_GOALS (both arbitrary)
    num_envs, num_sampled_ags = sampled_states.shape[:2]

    # score the sampled_ags using q values from exploration policy
    flattened_sampled_ags = sampled_states.reshape(num_envs * num_sampled_ags, -1)
    scores = -1 * self._compute_q_expl(flattened_sampled_ags)
    scores = scores.reshape(num_envs, num_sampled_ags)

    return scores

  def score_states(self, states):
      ag = states['achieved_goal']
      density_module = getattr(self, self.density_module)
      if not density_module.ready:
          # density_module._optimize(force=True)
        return np.zeros(ag.shape[0])
      states_score = -1 * density_module.evaluate_log_density(ag.astype(np.float32))
      if self.config.clip_density:
        states_score = np.clip(states_score, -self.config.clip_density, self.config.clip_density)

      return states_score

  def get_context(self, states):
      ag = states['achieved_goal']
      num_envs = ag.shape[0]

      density_module = getattr(self, self.density_module)
      if not density_module.ready:
          # density_module._optimize(force=True)
        return np.ones((num_envs, self.num_context)) / self.num_context

      ag_tile = np.tile(ag, (self.num_context, )).reshape(num_envs, self.num_context, -1)
      context_states = ag_tile + self.context_states
      flattened_context_states = context_states.reshape(num_envs * self.num_context, -1).astype(np.float32)
      density_context_states = softmax(density_module.evaluate_log_density(flattened_context_states)\
          .reshape(num_envs, self.num_context))
      density_context_states_normalized = density_context_states #/ np.linalg.norm(density_context_states, axis=-1, keepdims=True)
      return density_context_states_normalized

  def _compute_q_expl(self, numpy_states):
      states = self.torch(numpy_states)
      training = self.training
      self.eval_mode()
      with torch.no_grad():
          max_actions = self.expl_actor(states)
          if isinstance(max_actions, tuple):
              max_actions = max_actions[0]
          q_value = torch.min(self.expl_critic(states, max_actions), self.expl_critic2(states, max_actions))
      if training:
          self.train_mode()
      return self.numpy(q_value)