import gym
import yaml
from rl_games.torch_runner import Runner


def test1():

  ## walker_envpool config:
  walker_config = {'params': {'algo': {'name': 'a2c_continuous'},
  'config': {'bound_loss_type': 'regularisation',
   'bounds_loss_coef': 0.0005,
   'clip_value': True,
   'critic_coef': 4,
   'e_clip': 0.2,
   'entropy_coef': 0.0,
   'env_config': {'env_name': 'HalfCheetah-v4', 'seed': 5, 'exclude_current_positions_from_observation': False},
   'env_name': 'envpool',
   'gamma': 0.99,
   'grad_norm': 1.0,
   'horizon_length': 64,
   'kl_threshold': 0.008,
   'learning_rate': '3e-4',
   'lr_schedule': 'adaptive',
   'max_epochs': 5000,
   'mini_epochs': 5,
   'minibatch_size': 2048,
   'name': 'HalfCheetah-v4',
   'normalize_advantage': True,
   'normalize_input': True,
   'normalize_value': True,
   'num_actors': 64,
   'player': {'render': True},
   'ppo': True,
   'reward_shaper': {'scale_value': 5},
   'schedule_type': 'standard',
   'score_to_win': 20000,
   'tau': 0.95,
   'truncate_grads': True,
   'use_smooth_clamp': True,
   'value_bootstrap': True,
   'use_sea': True,
   "sea_config":{
      "objective_dim":64,
      "achievement_buffer_config":{
         "length":8,
         "capacity":4096,
         "minimum_length":2
      },
      "mini_epochs":1,
      "minibatch_size": 2048,
      "truncate_grads": False,
      "normalize_input": False,
      "learning_rate": 3e-4,
      'network': {'mlp': {'activation': 'elu',
         'initializer': {'name': 'default'},
         'units': [256, 256]},
         'name': 'actor_critic',
         'separate': False,
         'space': {'continuous': {'fixed_sigma': True,
         'mu_activation': 'None',
         'mu_init': {'name': 'default'},
         'sigma_activation': 'None',
         'sigma_init': {'name': 'const_initializer', 'val': 0}}}}
   }
   },
  'model': {'name': 'continuous_a2c_logstd'},
  'network': {'mlp': {'activation': 'elu',
    'initializer': {'name': 'default'},
    'units': [256, 128, 64]},
   'name': 'actor_critic',
   'separate': False,
   'space': {'continuous': {'fixed_sigma': True,
     'mu_activation': 'None',
     'mu_init': {'name': 'default'},
     'sigma_activation': 'None',
     'sigma_init': {'name': 'const_initializer', 'val': 0}}}},
  'seed': 5}}

  config = walker_config
  config['params']['config']['full_experiment_name'] = 'HalfCheetah_mujoco_f'
  config['params']['config']['max_epochs'] = 10000
  # config['params']['config']['save_best_after'] = 1
  # config['params']['config']['horizon_length'] = 512
  # config['params']['config']['num_actors'] = 8
  # config['params']['config']['minibatch_size'] = 1024

  runner = Runner()
  runner.load(config)
  runner.run({
      'train': True,
  })

  ## config from the openai gym mujoco (should have the same network and normalization) to render result:

  player_walker_config = {'params': {'algo': {'name': 'a2c_continuous'},
  'config': {'bounds_loss_coef': 0.0,
   'clip_value': False,
   'critic_coef': 1,
   'e_clip': 0.2,
   'entropy_coef': 0.0,
   'env_config': {'name': 'HalfCheetah-v3', 'seed': 5, 'exclude_current_positions_from_observation': False},
   'env_name': 'openai_gym',
   'gamma': 0.995,
   'grad_norm': 0.5,
   'horizon_length': 128,
   'kl_threshold': 0.008,
   'learning_rate': '3e-4',
   'lr_schedule': 'adaptive',
   'max_epochs': 5000,
   'mini_epochs': 4,
   'minibatch_size': 512,
   'name': 'Walker2d-v3',
   'normalize_advantage': True,
   'normalize_input': True,
   'normalize_value': True,
   'num_actors': 16,
   'player': {'render': True},
   'ppo': True,
   'reward_shaper': {'scale_value': 5},
   'schedule_type': 'standard',
   'score_to_win': 10000,
   'tau': 0.95,
   'truncate_grads': True,
   'value_bootstrap': True,
   'use_sea': True,
   "sea_config":{
      "objective_dim":64,
      "achievement_buffer_config":{
         "length":8,
         "capacity":512,
         "minimum_length":2
      },
      "mini_epochs":4,
      "minibatch_size":64,
      "truncate_grads": False,
      "normalize_input": False,
      "learning_rate": 3e-4,
      'network': {'mlp': {'activation': 'elu',
         'initializer': {'name': 'default'},
         'units': [256, 256]},
         'name': 'actor_critic',
         'separate': False,
         'space': {'continuous': {'fixed_sigma': True,
         'mu_activation': 'None',
         'mu_init': {'name': 'default'},
         'sigma_activation': 'None',
         'sigma_init': {'name': 'const_initializer', 'val': 0}}}}
   }
   },
  'model': {'name': 'continuous_a2c_logstd'},
  'network': {'mlp': {'activation': 'elu',
    'initializer': {'name': 'default'},
    'units': [256, 128, 64]},
   'name': 'actor_critic',
   'separate': False,
   'space': {'continuous': {'fixed_sigma': True,
     'mu_activation': 'None',
     'mu_init': {'name': 'default'},
     'sigma_activation': 'None',
     'sigma_init': {'name': 'const_initializer', 'val': 0}}}},
  'seed': 5}}

  config = player_walker_config
  config['params']['config']['player']['render'] = False
  config['params']['config']['player']['games_num'] = 5

  runner.load(config)
  agent = runner.create_player()
  agent.restore('runs/HalfCheetah_mujoco_f/nn/HalfCheetah-v4.pth')
  # agent.restore('notebooks/runs/HalfCheetah_mujoco_sea_double2/nn/last_HalfCheetah-v4_ep_500_rew__0.96640277_.pth')

  agent.run()

  print(len(agent.sea_embeds))


def test2():
  runner = Runner()

  config = yaml.safe_load(open('rl_games/configs/mujoco/ant_envpool.yaml'))
  config['params']['config']['env_name'] = 'openai_gym'
  config['params']['config']['env_config'] = {
      'name': 'Ant-v4',
      'render_mode': 'single_rgb_array',
      'seed': 5
  }
  config['params']['config']['player']['render'] = False
  config['params']['config']['player']['games_num'] = 1

  runner.load(config)
  agent = runner.create_player()
  agent.restore('runs/Ant-v4_envpool_13-19-43-35/nn/Ant-v4_envpool.pth')


if __name__ == "__main__":
  test1()