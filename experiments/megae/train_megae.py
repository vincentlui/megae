# 1. Imports
from mrl.import_all import *
from mrl.modules.train import debug_vectorized_experience
from experiments.mega.make_env import make_env
import time
import os
import gym
import numpy as np
import torch.nn as nn

from mrl.megae.curiosity import DensityMegaeCuriosity, DensityAndExplorationMegaeCuriosity
from mrl.megae.policy import ExplorationActorPolicy
from mrl.megae.train import MegaeTrain
from mrl.megae.algo import DDPG2, SAC2
from mrl.megae.config import megae_config, antconfig, fetchconfig, testconfig
from mrl.megae.replay_buffer import MegaeBuffer, Megae2Buffer
from mrl.megae.normalizer import Normalizer, MeanStdNormalizer
from mrl.megae.empowerment import InverseDynamicsNetwork, JSMIT, JSMI


# 2. Get default config and update any defaults (this automatically updates the argparse defaults)
# config = megae_config()

# 3. Make changes to the argparse below

def main(args, config):

  # 4. Update the config with args, and make the agent name. 
  if args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)

  merge_args_into_config(args, config)
  
  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1-config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(- args.env_max_step - 5, 2), 0.)

  if args.sparse_reward_shaping:
    config.clip_target_range = (-np.inf, np.inf)

  config.agent_name = make_agent_name(config, ['env','her','layers','seed','tb','ag_curiosity','eexplore', 'var_context'], prefix=args.prefix)

  # 5. Setup / add basic modules to the config
  config.update(
      dict(
          trainer=MegaeTrain(),
          evaluation=EpisodicEval(),
          policy=ExplorationActorPolicy(), #ActorPolicy(),
          policy_expl=ExplorationActorPolicy(),
          logger=Logger(),
          state_normalizer=Normalizer(MeanStdNormalizer()),
          replay=MegaeBuffer(module_name='replay_buffer'),
          # replay2=Megae2Buffer(module_name='replay_buffer_expl')
      ))

  state_normalizer2 = Normalizer(MeanStdNormalizer()) # Normalize context states
  state_normalizer2.module_name = 'state_normalizer_expl'
  config.state_normalizer2 = state_normalizer2

  if config.normalize_reward:
    reward_normalizer = Normalizer(MeanStdNormalizer())
    reward_normalizer.module_name = 'reward_normalizer'
    config.reward_normalizer = reward_normalizer

    if args.use_empowerment:
      reward_normalizer2 = Normalizer(MeanStdNormalizer())
      reward_normalizer2.module_name = 'reward_emp_normalizer'
      config.reward_normalizer2 = reward_normalizer2

  config.prioritized_mode = args.prioritized_mode
  if config.prioritized_mode == 'mep':
    config.prioritized_replay = EntropyPrioritizedOnlineHERBuffer()

  if not args.no_ag_kde:
    config.ag_kde = RawKernelDensity('ag', optimize_every=1, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
  if args.ag_curiosity is not None:
    config.dg_kde = RawKernelDensity('dg', optimize_every=500, samples=10000, kernel='tophat', bandwidth = 0.2)
    config.ag_kde_tophat = RawKernelDensity('ag', optimize_every=100, samples=10000, kernel='tophat', bandwidth = 0.2, tag='_tophat')
    if args.transition_to_dg:
      config.alpha_curiosity = CuriosityAlphaMixtureModule()
    if 'rnd' in args.ag_curiosity:
      config.ag_rnd = RandomNetworkDensity('ag')
    if 'flow' in args.ag_curiosity:
      config.ag_flow = FlowDensity('ag')

    use_qcutoff = not args.no_cutoff

    if args.ag_curiosity == 'minq':
      config.ag_curiosity = QAchievedGoalCuriosity(max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randq':
      config.ag_curiosity = QAchievedGoalCuriosity(max_steps = args.env_max_step, randomize=True, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'megaekde':
      config.ag_curiosity = DensityMegaeCuriosity(max_steps=args.env_max_step,
                                                   num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff,
                                                   keep_dg_percent=args.keep_dg_percent,
                                                  exploration_percent=args.exploration_percent,
                                                  num_context=args.num_context,
                                                  context_var=args.var_context,
                                                  context_dist=args.context_dist,
                                                  initial_explore_percent=args.init_explore_percent
                                                  )
    elif args.ag_curiosity == 'megaerandkde':
      config.ag_curiosity = DensityMegaeCuriosity(max_steps=args.env_max_step,
                                                   num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff,
                                                   keep_dg_percent=args.keep_dg_percent,
                                                  randomize=True,
                                                  alpha=args.alpha,
                                                  exploration_percent=args.exploration_percent,
                                                  num_context=args.num_context,
                                                  context_var=args.var_context,
                                                  context_dist=args.context_dist,
                                                  initial_explore_percent=args.init_explore_percent
                                                  )
    elif args.ag_curiosity == 'megaeflow':
      config.ag_curiosity = DensityMegaeCuriosity('ag_flow', max_steps=args.env_max_step,
                                                   num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff,
                                                   keep_dg_percent=args.keep_dg_percent,
                                                  exploration_percent=args.exploration_percent,
                                                  num_context=args.num_context,
                                                  context_var=args.var_context,
                                                  context_dist=args.context_dist,
                                                  initial_explore_percent=args.init_explore_percent
                                                  )
    elif args.ag_curiosity == 'empkde':
      config.ag_curiosity = DensityAndExplorationMegaeCuriosity('ag_kde', max_steps=args.env_max_step,
                                                  density_percent=args.density_percent,
                                                   num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff,
                                                   keep_dg_percent=args.keep_dg_percent,
                                                  exploration_percent=args.exploration_percent,
                                                  num_context=args.num_context,
                                                  context_var=args.var_context,
                                                  context_dist=args.context_dist,
                                                  initial_explore_percent=args.init_explore_percent
                                                  )
    elif args.ag_curiosity == 'minrnd':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_rnd', max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minflow':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_flow', max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(alpha = args.alpha, max_steps = args.env_max_step, randomize=True, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randrnd':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_rnd', alpha = args.alpha, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randflow':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_flow', alpha = args.alpha, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'goaldisc':
      config.success_predictor = GoalSuccessPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe)
      config.ag_curiosity = SuccessAchievedGoalCuriosity(max_steps=args.env_max_step, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'entropygainscore':
      config.bg_kde = RawKernelDensity('bg', optimize_every=args.env_max_step, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
      config.bgag_kde = RawJointKernelDensity(['bg','ag'], optimize_every=args.env_max_step, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
      config.ag_curiosity = EntropyGainScoringGoalCuriosity(max_steps=args.env_max_step, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    else:
      raise NotImplementedError

  if args.noise_type.lower() == 'gaussian': noise_type = GaussianProcess
  if args.noise_type.lower() == 'ou': noise_type = OrnsteinUhlenbeckProcess
  config.action_noise = ContinuousActionNoise(noise_type, std=ConstantSchedule(args.action_noise))

  if args.alg.lower() == 'ddpg': 
    config.algorithm1 = DDPG2('algorithm1', optimize_every=1, actor_name='actor', critic_name='critic', clip_target=True)
  elif args.alg.lower() == 'sac':
    config.algorithm1 = SAC2('algorithm1', optimize_every=1, actor_name='actor', critic_name='critic', clip_target=True)
  else:
    raise NotImplementedError

  if args.alg_expl.lower() == 'ddpg':
    config.algorithm2 = DDPG2('algorithm2', optimize_every=1, actor_name='expl_actor', critic_name='expl_critic', is_explore=True)
  elif args.alg_expl.lower() == 'sac':
    config.algorithm2 = SAC2('algorithm2', optimize_every=1, actor_name='expl_actor', critic_name='expl_critic', is_explore=True)
  else:
    raise NotImplementedError

  # 6. Setup / add the environments and networks (which depend on the environment) to the config
  env, eval_env = make_env(args)
  if args.first_visit_done:
    env1, eval_env1 = env, eval_env
    env = lambda: FirstVisitDoneWrapper(env1())
    eval_env = lambda: FirstVisitDoneWrapper(eval_env1())
  if args.first_visit_succ:
    config.first_visit_succ = True

  config.train_env = EnvModule(env, num_envs=args.num_envs, seed=args.seed)
  config.eval_env = EnvModule(eval_env, num_envs=args.num_eval_envs, name='eval_env', seed=args.seed + 1138)

  e = config.eval_env
  config.critic = PytorchModel('critic',
                               lambda: Critic(
                                 FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm,
                                        make_activ(config.activ)), 1))
  if args.alg.lower() == 'ddpg':
    config.actor = PytorchModel('actor',
                                lambda: Actor(
                                  FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)),
                                  e.action_dim, e.max_action))
  elif args.alg.lower() == 'sac':
    config.actor = PytorchModel('actor',
                                lambda: StochasticActor(
                                  FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)),
                                  e.action_dim, e.max_action))
    config.critic2 = PytorchModel('critic2',
                                  lambda: Critic(
                                    FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm,
                                           make_activ(config.activ)), 1))
  else:
    raise NotImplementedError

  config.expl_critic = PytorchModel('expl_critic',
                                    lambda: Critic(
                                      FCBody(e.state_dim + args.num_context + e.action_dim, args.layers, nn.LayerNorm,
                                             make_activ(config.activ)), 1))
  if args.alg_expl.lower() == 'ddpg':
    config.expl_actor = PytorchModel('expl_actor',
                                     lambda: Actor(
                                       FCBody(e.state_dim + args.num_context, args.layers, nn.LayerNorm,
                                              make_activ(config.activ)),
                                       e.action_dim, e.max_action))
  elif args.alg_expl.lower() == 'sac':
    config.expl_actor = PytorchModel('expl_actor',
                                     lambda: StochasticActor(
                                       FCBody(e.state_dim + args.num_context, args.layers, nn.LayerNorm,
                                              make_activ(config.activ)),
                                       e.action_dim, e.max_action))
    config.expl_critic2 = PytorchModel('expl_critic2',
                                         lambda: Critic(
                                           FCBody(e.state_dim + args.num_context + e.action_dim, args.layers,
                                                  nn.LayerNorm,
                                                  make_activ(config.activ)), 1))
  else:
    raise NotImplementedError

  if args.ag_curiosity == 'goaldisc':
    config.goal_discriminator = PytorchModel('goal_discriminator', lambda: Critic(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))

  if args.reward_module == 'env':
    config.goal_reward = GoalEnvReward()
  elif args.reward_module == 'intrinsic':
    config.goal_reward = NeighborReward()
    config.neighbor_embedding_network = PytorchModel('neighbor_embedding_network',
                                                     lambda: FCBody(e.goal_dim, (256, 256)))
  else:
    raise ValueError('Unsupported reward module: {}'.format(args.reward_module))

  # Empowerment
  if args.use_empowerment:
    config.empowerment_net = PytorchModel('T',
                                        lambda: JSMIT(FCBody(e.state_dim + args.num_context + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm,
                                             make_activ(config.activ))))
    config.empowerment = JSMI(config.expl_actor, optimize_every=1)

  if config.eval_env.goal_env:
    if not (args.first_visit_done or args.first_visit_succ):
      config.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done


  # 7. Make the agent and run the training loop.
  agent = mrl.config_to_agent(config)

  if args.visualize_trained_agent:
    print(agent.ag_curiosity.context_states)
    agent.config.device = 'cpu'
    # agent.config.go_eexplore=0.1
    agent.eexplore = 0.0001
    print("Loading agent at epoch {}".format(0))
    agent.load('checkpoint')
    
    if args.intrinsic_visualization:
      agent.eval_mode()
      agent.train(10000, render=True, dont_optimize=True)

    else:
      agent.eval_mode()
      env = agent.eval_env

      for _ in range(10000):
        print("NEW EPISODE")
        agent.ag_curiosity.go_explore = np.zeros_like(agent.ag_curiosity.go_explore)
        state = env.reset()
        env.render()
        done = False
        context = agent.ag_curiosity.get_context(state)
        reward = [-1.]
        explore = False
        while not done:
          time.sleep(0.02)
          if reward[0] > -0.5:
            explore = True
          if explore:
            agent.policy.training = True
            # agent.ag_curiosity.go_explore += 1.
            # action = agent.policy(state, greedy=False)
            action = agent.policy(state, greedy=True, context=context, is_explore=np.array([1.]))
          else:
            agent.policy.training = False
            action = agent.policy(state)
          state, reward, done, info = env.step(action)
          env.render()
          context = agent.ag_curiosity.get_context(state)
          # print(reward[0])
  else:
    ag_buffer = agent.replay_buffer.buffer.BUFF.buffer_ag
    bg_buffer = agent.replay_buffer.buffer.BUFF.buffer_bg

    # EVALUATE
    res = np.mean(agent.eval(num_episodes=30).rewards)
    agent.logger.log_color('Initial test reward (30 eps):', '{:.2f}'.format(res))
    render = False
    for epoch in range(int(args.max_steps // args.epoch_len)):
      t = time.time()
      agent.train(num_steps=args.epoch_len, render=render)

      # VIZUALIZE GOALS
      if args.save_embeddings:
        # sample_idxs = np.random.choice(len(ag_buffer), size=min(len(ag_buffer), args.epoch_len), replace=False)
        last_idxs = np.arange(max(0, len(ag_buffer)-args.epoch_len), len(ag_buffer))
        # agent.logger.add_embedding('rand_ags', ag_buffer.get_batch(sample_idxs))
        agent.logger.add_embedding('last_ags', ag_buffer.get_batch(last_idxs))
        agent.logger.add_embedding('last_bgs', bg_buffer.get_batch(last_idxs))

      # EVALUATE
      res = np.mean(agent.eval(num_episodes=30).rewards)
      agent.logger.log_color('Test reward (30 eps):', '{:.2f}'.format(res))
      agent.logger.log_color('Epoch time:', '{:.2f}'.format(time.time() - t), color='yellow')

      print("Saving agent at epoch {}".format(epoch))
      agent.save('checkpoint')
      if args.save_gap is not None and epoch % args.save_gap == 0:
        agent.save_gap(epoch, subfolder='checkpoint')

def get_config(config_name):
  if config_name is not None:
    if config_name.lower() == 'ant':
      config2 = antconfig()
    elif config_name.lower() == 'fetch':
      config2 = fetchconfig()
    elif config_name.lower() == 'test':
      config2 = testconfig()
    else:
      raise NotImplementedError
  else:
    config2 = megae_config()
  return config2


# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train DDPG", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--use_config', default=None, type=str, help='which config to use: {ant, fetch}')
  config_arg = parser.parse_known_args()
  config = get_config(config_arg[0].use_config)
  parser.add_argument('--parent_folder', default='/data/bing/lui/log/megae', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='proto', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--save_gap', default=None, type=int, help="Save every n epochs")
  parser.add_argument('--env', default="FetchPush-v1", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=5000000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='DDPG', type=str, help='algorithm to use (DDPG or SAC)')
  parser.add_argument('--alg_expl', default='SAC', type=str, help='algorithm to use (DDPG or SAC)')
  parser.add_argument(
      '--layers', nargs='+', default=(512,512,512), type=int, help='sizes of layers for actor/critic networks')
  parser.add_argument('--noise_type', default='Gaussian', type=str, help='type of action noise (Gaussian or OU)')
  parser.add_argument('--tb', default='', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  # parser.add_argument('--num_envs', default=1, type=int, help='number of envs')

  # Make env args
  parser.add_argument('--eval_env', default='', type=str, help='evaluation environment')
  parser.add_argument('--test_with_internal', default=True, type=bool, help='test with internal reward fn')
  parser.add_argument('--reward_mode', default=0, type=int, help='reward mode')
  parser.add_argument('--env_max_step', default=50, type=int, help='max_steps_env_environment')
  parser.add_argument('--per_dim_threshold', default='0.', type=str, help='per_dim_threshold')
  parser.add_argument('--hard', action='store_true', help='hard mode: all goals are high up in the air')
  parser.add_argument('--pp_in_air_percentage', default=0.5, type=float, help='sets in air percentage for fetch pick place')
  parser.add_argument('--pp_min_air', default=0.2, type=float, help='sets the minimum height in the air for fetch pick place when in hard mode')
  parser.add_argument('--pp_max_air', default=0.45, type=float, help='sets the maximum height in the air for fetch pick place')
  parser.add_argument('--train_dt', default=0., type=float, help='training distance threshold')
  parser.add_argument('--slow_factor', default=1., type=float, help='slow factor for moat environment; lower is slower. ')

  # Other args
  parser.add_argument('--first_visit_succ', action='store_true', help='Episodes are successful on first visit (soft termination).')
  parser.add_argument('--first_visit_done', action='store_true', help='Episode terminates upon goal achievement (hard termination).')
  parser.add_argument('--ag_curiosity', default=None, help='the AG Curiosity model to use: {minq, randq, minkde}')
  parser.add_argument('--bandwidth', default=0.1, type=float, help='bandwidth for KDE curiosity')
  parser.add_argument('--kde_kernel', default='gaussian', type=str, help='kernel for KDE curiosity')
  parser.add_argument('--num_sampled_ags', default=100, type=int, help='number of ag candidates sampled for curiosity')
  parser.add_argument('--alpha', default=-1.0, type=float, help='Skewing parameter on the empirical achieved goal distribution. Default: -1.0')
  parser.add_argument('--reward_module', default='env', type=str, help='Reward to use (env or intrinsic)')
  parser.add_argument('--save_embeddings', action='store_true', help='save ag embeddings during training?')
  parser.add_argument('--succ_bs', default=100, type=int, help='success predictor batch size')
  parser.add_argument('--succ_hl', default=200, type=int, help='success predictor history length')
  parser.add_argument('--succ_oe', default=250, type=int, help='success predictor optimize every')
  parser.add_argument('--ag_pred_ehl', default=5, type=int, help='achieved goal predictor number of timesteps from end to consider in episode')
  parser.add_argument('--transition_to_dg', action='store_true', help='transition to the dg distribution?')
  parser.add_argument('--no_cutoff', action='store_true', help="don't use the q cutoff for curiosity")
  parser.add_argument('--visualize_trained_agent', action='store_true', help="visualize the trained agent")
  parser.add_argument('--intrinsic_visualization', action='store_true', help="if visualized agent should act intrinsically; requires saved replay buffer!")
  parser.add_argument('--keep_dg_percent', default=-1e-1, type=float, help='Percentage of time to keep desired goals')
  parser.add_argument('--prioritized_mode', default='none', type=str, help='Modes for prioritized replay: none, mep (default: none)')
  parser.add_argument('--no_ag_kde', action='store_true', help="don't track ag kde")

  # Megae args
  parser.add_argument('--num_context', default=20, type=int, help='number of context states density for exploration policy')
  parser.add_argument('--var_context', default=1., type=float, help='variance of context states')
  parser.add_argument('--exploration_percent', default=0.8, type=float, help='percentage of exploration steps')
  parser.add_argument('--context_dist', default='normal', type=str, help='percentage of exploration steps')
  parser.add_argument('--init_explore_percent', default=0., type=float, help='percentage of pure exploration at start')
  parser.add_argument('--use_empowerment', action='store_true', help='Include empowerment as intrinsic reward')
  parser.add_argument('--density_percent', default='0.', type=float, help='percentage of goals using density objective')

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(args, config)
