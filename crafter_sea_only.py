import gym
import yaml
from rl_games.torch_runner import Runner


def test1():
   
   config = yaml.safe_load(open('rl_games/configs/ppo_crafter.yaml'))
   config['params']['config']['full_experiment_name'] = 'crafter_sea_only_res+no_mlp_optim_coef'
   config['params']['config']['no_train_actor_critic'] = True
   config['params']['config']['no_load_sea'] = True
   config['params']['config']['only_load_weights'] = True
   config['params']['config']['save_frequency'] = 10
   config['params']['config']['sea_config']['contrast_coef'] = 128 * 20

   runner = Runner()
   runner.load(config)
   runner.run({
      'train': True,
      'checkpoint': 'runs/Crafter_17-17-50-08/nn/Crafter.pth'
   })


def save():
   
   config = yaml.safe_load(open('rl_games/configs/ppo_crafter.yaml'))
   config['params']['config']['full_experiment_name'] = 'crafter_save'
   config['params']['config']['max_epochs'] = 1
   config['params']['config']['save_best_after'] = 100
   config['params']['config']['horizon_length'] = 256
   config['params']['config']['num_actors'] = 32
   config['params']['config']['minibatch_size'] = 1

   config['params']['config']['sea_config']['mini_epochs'] = 1
   config['params']['config']['sea_config']['train'] = False
   config['params']['config']['sea_config']['save_achievement_buffer'] = True
   # config['params']['config']['sea_config']['achievement_buffer_checkpoint'] = 'runs/HalfCheetah_mujoco_sea_test_2/nn/achievement_buffer.pth'
   config['params']['config']['sea_config']['achievement_buffer_config']['capacity'] = 2000

   config['params']['config']['no_train_actor_critic'] = True
   config['params']['config']['no_load_sea'] = True
   config['params']['config']['only_load_weights'] = True
   config['params']['config']['save_frequency'] = 10
   config['params']['config']['sea_config']['contrast_coef'] = 128 * 20

   runner = Runner()
   runner.load(config)
   runner.run({
      'train': True,
      'checkpoint': 'runs/Crafter_17-17-50-08/nn/Crafter.pth'
   })


if __name__ == "__main__":
   save()