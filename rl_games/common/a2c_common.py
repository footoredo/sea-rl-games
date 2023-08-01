from asyncore import write
import copy
import os

from rl_games.common import vecenv

from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from rl_games.common.experience import ExperienceBuffer, AchievementBuffer, SEABuffer
from rl_games.common.interval_summary_writer import IntervalSummaryWriter
from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics
from rl_games.algos_torch import  model_builder
from rl_games.interfaces.base_algorithm import  BaseAlgorithm
import numpy as np
import time
import gym

from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
from torch import nn
import torch.distributed as dist
 
from time import sleep

from rl_games.common import common_losses


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def get_dict_losses(losses):
    if len(losses) > 0 and type(losses[0]) == dict:
        dict_losses = dict()
        for loss in losses:
            for key, value in loss.items():
                if key not in dict_losses:
                    dict_losses[key] = []
                dict_losses[key].append(value)
        return dict_losses
    else:
        return losses
    

def write_dict_stats(writer, name, stats, frame):
    # print(stats)
    if type(stats) == dict:
        for key, _stats in stats.items():
            writer.add_scalar(f'{name}/{key}', torch_ext.mean_list(_stats).item(), frame)
    else:
        writer.add_scalar(f'{name}', torch_ext.mean_list(stats).item(), frame)


def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}')
        elif max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}')


class A2CBase(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'

        # This helps in PBT when we need to restart an experiment with the exact same name, rather than
        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")

        self.config = config
        self.use_rnd = config.get('use_rnd', False)
        self.use_sea = config.get('use_sea', False)
        self.sea_config = config.get('sea_config', {})
        self.sea_coef = self.sea_config.get('coef', 1.)
        self.no_train_actor_critic = config.get('no_train_actor_critic', False)
        self.no_load_sea = config.get('no_load_sea', False)
        self.only_load_weights = config.get('only_load_weights', False)
        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.load_networks(params)
        self.multi_gpu = config.get('multi_gpu', False)
        self.rank = 0
        self.rank_size = 1
        self.curr_frames = 0

        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)

            self.device_name = 'cuda:' + str(self.rank)
            config['device'] = self.device_name
            if self.rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

        self.use_diagnostics = config.get('use_diagnostics', False)

        if self.use_diagnostics and self.rank == 0:
            self.diagnostics = PpoDiagnostics()
        else:
            self.diagnostics = DefaultDiagnostics()

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        self.vec_env = None
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, self.use_sea, self.sea_config, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = config.get('vec_env', None)

        self.ppo_device = config.get('device', 'cuda:0')
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.truncate_grads = self.config.get('truncate_grads', False)

        if self.has_central_value:
            self.state_space = self.env_info.get('state_space', None)
            if isinstance(self.state_space,gym.spaces.Dict):
                self.state_shape = {}
                for k,v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        print('save_freq', self.save_freq)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')

        # Setting learning rate scheduler
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)

        elif self.linear_lr:
            
            if self.max_epochs == -1 and self.max_frames == -1:
                print("Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one.")
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.max_epochs

                if self.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.max_frames

                self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']), 
                    max_steps = max_steps,
                    use_epochs = use_epochs, 
                    apply_to_entropy = config.get('schedule_entropy', False),
                    start_entropy_coef = config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.bptt_len = self.config.get('bptt_length', self.seq_len) # not used right now. Didn't show that it is usefull
        self.zero_rnn_on_done = self.config.get('zero_rnn_on_done', True)
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 2000)
        print('current training device:', self.ppo_device)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        if self.use_sea:
            self.game_achv_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
            self.game_env_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
            self.game_true_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
            self.achievement_completions = dict()
            self.objective_rewards = dict()
        self.obs = None
        self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']

        if self.rank == 0:
            writer = SummaryWriter(self.summaries_dir)
            if self.population_based_training:
                self.writer = IntervalSummaryWriter(writer, self.config)
            else:
                self.writer = writer
        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')
        self.use_smooth_clamp = self.config.get('use_smooth_clamp', False)

        if self.use_smooth_clamp:
            self.actor_loss_func = common_losses.smoothed_actor_loss
        else:
            self.actor_loss_func = common_losses.actor_loss

        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5) #'0.25'
            self.advantage_mean_std = GeneralizedMovingStats((1,), momentum=momentum).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None
        self.last_state_indices = None

        #self_play
        if self.has_self_play_config:
            print('Initializing SelfPlay Manager')
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)

        # features
        self.algo_observer = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.has_soft_aug = self.soft_aug is not None
        # soft augmentation not yet supported
        assert not self.has_soft_aug

    def trancate_gradients_and_step(self):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.rank_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            if self.multi_optimizer:
                for i in range(len(self.optimizers)):
                    self.scaler.unscale_(self.optimizers[i])
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        if self.multi_optimizer:
            for i, optimizer in enumerate(self.optimizers):
                has_grad = False
                for param in self.model.a2c_network.nets[i].parameters():
                    if param.grad is not None and torch.norm(param.grad) > 1e-4:
                        has_grad = True
                        break
                if has_grad:
                    self.scaler.step(optimizer)
        else:
            self.scaler.step(self.optimizer)
        self.scaler.update()

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        network_params = copy.copy(params['network'])
        self.config['network'] = builder.load(params)
        has_central_value_net = self.config.get('central_value_config') is not  None
        if has_central_value_net:
            print('Adding Central Value Network')
            if 'model' not in params['config']['central_value_config']:
                params['config']['central_value_config']['model'] = {'name': 'central_value'}
            network = builder.load(params['config']['central_value_config'])
            self.config['central_value_config']['network'] = network
        if self.use_sea:
            print('Adding SEA net')
            if 'model' not in params['config']['sea_config']:
                params['config']['sea_config']['model'] = {'name': 'sea'}
            if 'network' not in params['config']['sea_config']:
                params['config']['sea_config']['network'] = network_params
            network = builder.load(params['config']['sea_config'])
            self.config['sea_config']['network'] = network

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, rnd_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)

        if not self.no_train_actor_critic:
            write_dict_stats(self.writer, "losses/a_loss", get_dict_losses(a_losses), frame)
            write_dict_stats(self.writer, "losses/c_loss", get_dict_losses(c_losses), frame)
            write_dict_stats(self.writer, "losses/rnd_loss", get_dict_losses(rnd_losses), frame)
            write_dict_stats(self.writer, "losses/entropy", get_dict_losses(entropies), frame)
            # self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
            # self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)
            # self.writer.add_scalar('losses/rnd_loss', torch_ext.mean_list(rnd_losses).item(), frame)
            # self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)

            self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
            self.writer.add_scalar('info/lr_mul', lr_mul, frame)
            self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
            self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)

        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        self.model.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        if self.multi_optimizer:
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
            if 'rnd_gt' in res_dict and res_dict['rnd_gt'] is not None:
                rnd_gt = res_dict.pop('rnd_gt')
                rnd_pred = res_dict.pop('rnd_pred')
                # print(rnd_gt.shape, rnd_pred.shape)
                res_dict['rnd_error'] = torch.square(rnd_pred - rnd_gt).sum(-1) / np.sqrt(rnd_pred.shape[-1])
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states
                }
                result = self.model(input_dict)
                value = result['values']
            return value

    @property
    def device(self):
        return self.ppo_device

    def reset_envs(self):
        self.obs = self.env_reset()

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)
        if self.use_sea:
            self.achievement_buffer = AchievementBuffer(self.num_actors,
                                                        self.env_info["observation_space"]["observation"].shape,
                                                        self.env_info["action_space"].shape, 
                                                        device=self.ppo_device,
                                                        **self.sea_config["achievement_buffer_config"])
            self.sea_buffer = SEABuffer(self.env_info["observation_space"]["observation"].shape,
                                        self.env_info["action_space"].shape, 
                                        device=self.ppo_device,
                                        **self.sea_config["achievement_buffer_config"])
            self.save_achievement_buffer = self.sea_config.get('save_achievement_buffer', False)
            achievement_buffer_checkpoint = self.sea_config.get('achievement_buffer_checkpoint', None)
            if achievement_buffer_checkpoint:
                self.achievement_buffer.restore(achievement_buffer_checkpoint)
            self.sea_train = self.sea_config.get('train', True)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.current_achievements = [set() for _ in range(self.batch_size)]
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        if self.use_sea:
            self.current_achv_rewards = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.ppo_device)
            self.current_env_rewards = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.ppo_device)
            self.current_true_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
            self.true_dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_len
            assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

    def init_rnn_from_model(self, model):
        self.is_rnn = self.model.is_rnn()

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        return obs

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma * self.tau * nextnonterminal * lastgaelam) * masks_t
        return mb_advs

    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        self.game_rewards.clear()
        self.game_shaped_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def update_epoch(self):
        pass

    def train(self):
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()
    
    def train_sea(self):
        if self.sea_train:
            return self.sea_net.train_net(self.achievement_buffer, self.sea_buffer, self.frame)

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        if self.multi_optimizer:
            for i, optimizer in enumerate(self.optimizers):
                state[f'optimizer_{i}'] = optimizer.state_dict()
        else:
            state['optimizer'] = self.optimizer.state_dict()
        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        if self.use_sea:
            state['sea_net'] = self.sea_net.state_dict()
        state['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights):
        self.set_weights(weights)

        if not self.only_load_weights:
            self.epoch_num = weights['epoch'] # frames as well?
            self.frame = weights.get('frame', 0)
            if self.multi_optimizer:
                for i in range(len(self.optimizers)):
                    self.optimizers[i].load_state_dict(weights[f'optimizer_{i}'])
            else:
                self.optimizer.load_state_dict(weights['optimizer'])
            self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

            env_state = weights.get('env_state', None)

            if self.vec_env is not None:
                self.vec_env.set_env_state(env_state)

        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        if self.use_sea and not self.no_load_sea:
            if 'sea_net' not in weights:
                print('no sea_net weights found in checkpoint!')
            else:
                self.sea_net.load_state_dict(weights['sea_net'])


    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        if self.has_central_value:
            state['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights):
        if self.normalize_rms_advantage:
            self.advantage_mean_std.load_state_dic(weights['advantage_mean_std'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            self.model.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.set_stats_weights(weights)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k,v in obs_batch.items():
                if v.dtype == torch.uint8 and len(v.shape) >= 3:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            if self.use_sea:
                last_obs = self.obs['obs']['observation'].clone()
                last_objective = self.obs['obs']['objective'].clone()

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            if self.use_sea:
                # self.true_dones = torch.zeros_like(self.dones)
                for i in range(rewards.shape[0]):
                    self.sea_buffer.add(last_obs[i], res_dict['actions'][i], self.obs['obs']['observation'][i], rewards[i])
                    if rewards[i] > 0.5:
                        self.achievement_buffer.add(i, last_obs[i], res_dict['actions'][i], self.obs['obs']['observation'][i])
                    if infos["env_is_done"][i]:
                        self.achievement_buffer.episode_done(i)
                    self.true_dones[i] = int(infos["env_is_done"][i])
                    self.current_achievements[i] |= set(infos["step_completed"][i])

            step_time += (step_time_end - step_time_start)

            if 'rnd_error' in res_dict:
                # print(rewards.shape, res_dict['rnd_error'].shape)
                # print(res_dict['rnd_error'][:10])
                # obs_diff = (self.obs['obs']['observation'] - last_obs).square().sum(1)
                rewards += res_dict['rnd_error'].unsqueeze(-1) * self.rnd_reward_coef
                # rewards += obs_diff.unsqueeze(-1) * self.rnd_reward_coef

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
     
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            if self.use_sea:
                self.current_achv_rewards += rewards
                # print(infos[0]["env_reward"], len(infos), self.current_env_rewards.shape)
                # self.current_env_rewards[:, 0] += torch.tensor(infos["env_reward"], dtype=torch.float32, device=rewards.device)
                self.current_env_rewards[:, 0] += infos["env_reward"].clone().detach()
                self.current_true_lengths += 1
                true_done_indices = self.true_dones.nonzero(as_tuple=False)

                self.game_achv_rewards.update(self.current_achv_rewards[true_done_indices])
                self.game_env_rewards.update(self.current_env_rewards[true_done_indices])
                self.game_true_lengths.update(self.current_true_lengths[true_done_indices])

                true_not_dones = 1.0 - self.true_dones.float()

                self.current_achv_rewards = self.current_achv_rewards * true_not_dones.unsqueeze(1)
                self.current_env_rewards = self.current_env_rewards * true_not_dones.unsqueeze(1)
                self.current_true_lengths = self.current_true_lengths * true_not_dones

                for i in range(rewards.shape[0]):
                    if self.dones[i]:
                        objective = last_objective[i].nonzero().item()
                        named_objective = self.vec_env.objective_selector.known_objectives[objective]
                        if named_objective not in self.objective_rewards:
                            self.objective_rewards[named_objective] = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
                        self.objective_rewards[named_objective].update(torch.ones((1,)) * self.current_rewards[i].item())
                    if self.true_dones[i]:
                        # if len(self.current_achievements[i]) > 0:
                        #     print(self.current_achievements)
                        for achv in self.current_achievements[i]:
                            if achv not in self.achievement_completions:
                                self.achievement_completions[achv] = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
                            self.achievement_completions[achv].update(torch.ones((1,)))
                        for achv in self.achievement_completions.keys():
                            if achv not in self.current_achievements[i]:
                                self.achievement_completions[achv].update(torch.zeros((1,)))
                        self.current_achievements[i] = set()
            
            not_dones = 1.0 - self.dones.float()
            
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

    def play_steps_rnn(self):
        update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_len == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_len,:,:,:] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            if self.use_sea:
                last_obs = self.obs['obs']['observation'].clone()
                last_objective = self.obs['obs']['objective'].clone()

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            if self.use_sea:
                for i in range(rewards.shape[0]):
                    self.sea_buffer.add(last_obs[i], res_dict['actions'][i], self.obs['obs']['observation'][i], rewards[i])
                    if rewards[i] > 0.5:
                        self.achievement_buffer.add(i, last_obs[i], res_dict['action'][i], self.obs['obs']['observation'][i])
                    if infos["env_is_done"][i]:
                        self.achievement_buffer.episode_done(i)
                    self.true_dones[i] = int(infos["env_is_done"][i])
                    self.current_achievements[i] |= set(infos["step_completed"][i])

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            if len(all_done_indices) > 0:
                if self.zero_rnn_on_done:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            if self.use_sea:
                self.current_achv_rewards += rewards
                # print(infos[0]["env_reward"], len(infos), self.current_env_rewards.shape)
                self.current_env_rewards[:, 0] += torch.tensor(infos["env_reward"], dtype=torch.float32, device=rewards.device)
                self.current_true_lengths += 1
                true_done_indices = self.true_dones.nonzero(as_tuple=False)

                self.game_achv_rewards.update(self.current_achv_rewards[true_done_indices])
                self.game_env_rewards.update(self.current_env_rewards[true_done_indices])
                self.game_true_lengths.update(self.current_true_lengths[true_done_indices])

                true_not_dones = 1.0 - self.true_dones.float()

                self.current_achv_rewards = self.current_achv_rewards * true_not_dones.unsqueeze(1)
                self.current_env_rewards = self.current_env_rewards * true_not_dones.unsqueeze(1)
                self.current_true_lengths = self.current_true_lengths * true_not_dones

                for i in range(rewards.shape[0]):
                    if self.dones[i]:
                        objective = last_objective[i].nonzero().item()
                        named_objective = self.vec_env.objective_selector.known_objectives[objective]
                        if named_objective not in self.objective_rewards:
                            self.objective_rewards[named_objective] = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
                        self.objective_rewards[named_objective].update(torch.ones((1,)) * self.current_rewards[i].item())
                    if self.true_dones[i]:
                        for achv in self.current_achievements[i]:
                            if achv not in self.achievement_completions:
                                self.achievement_completions[achv] = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
                            self.achievement_completions[achv].update(torch.ones((1,)))
                        for achv in self.achievement_completions.keys():
                            if achv not in self.current_achievements[i]:
                                self.achievement_completions[achv].update(torch.zeros((1,)))
                        self.current_achievements[i] = set()

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()

        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1,2,0,3).reshape(-1,t_size, h_size))
        batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time
        return batch_dict


class DiscreteA2CBase(A2CBase):

    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)
    
        batch_size = self.num_agents * self.num_actors
        action_space = self.env_info['action_space']
        if type(action_space) is gym.spaces.Discrete:
            self.actions_shape = (self.horizon_length, batch_size)
            self.actions_num = action_space.n
            self.is_multi_discrete = False
        if type(action_space) is gym.spaces.Tuple:
            self.actions_shape = (self.horizon_length, batch_size, len(action_space)) 
            self.actions_num = [action.n for action in action_space]
            self.is_multi_discrete = True
        self.is_discrete = True

    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values']
        if self.use_action_masks:
            self.update_list += ['action_masks']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        self.vec_env.objective_selector.print()

        self.set_train()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses = []
        c_losses = []
        entropies = []
        kls = []
        if self.has_central_value:
            self.train_central_value()
        
        if self.use_sea:
            self.train_sea()

        if not self.no_train_actor_critic:
            for mini_ep in range(0, self.mini_epochs_num):
                ep_kls = []
                for i in range(len(self.dataset)):
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(self.dataset[i])
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    ep_kls.append(kl)
                    entropies.append(entropy)

                av_kls = torch_ext.mean_list(ep_kls)
                if self.multi_gpu:
                    dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                    av_kls /= self.rank_size

                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)
                kls.append(av_kls)
                self.diagnostics.mini_epoch(self, mini_ep)
                if self.normalize_input:
                    self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch
        else:
            last_lr = 0
            lr_mul = 0

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        rnn_masks = batch_dict.get('rnn_masks', None)
        returns = batch_dict['returns']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        dones = batch_dict['dones']
        rnn_states = batch_dict.get('rnn_states', None)
        objectives = batch_dict['obses'].get('objective', None)
        
        obses = batch_dict['obses']
        advantages = returns - values

        if self.normalize_value:
            # self.value_mean_std.train()
            # values = self.value_mean_std(values)
            # returns = self.value_mean_std(returns)
            # self.value_mean_std.eval()
            values = self.model.norm_value(values, objectives)
            returns = self.model.norm_value(returns, objectives)
            
        
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    if objectives is not None:
                        for i in range(objectives.shape[1]):
                            indices = objectives[:, i].nonzero()
                            if indices.shape[0] > 0:
                                adv = advantages[indices]
                                advantages[indices] = (adv - adv.mean()) / (adv.std() + 1e-8)
                    else:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks

        if self.use_action_masks:
            dataset_dict['action_masks'] = batch_dict['action_masks']

        self.dataset.update_values_dict(dataset_dict)
        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['dones'] = dones
            dataset_dict['obs'] = batch_dict['states'] 
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint
        self.obs = self.env_reset()

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, 
                                scaled_time, scaled_play_time, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)


                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.use_sea:
                        if self.game_achv_rewards.current_size > 0:
                            mean_achv_rewards = self.game_achv_rewards.get_mean()
                            mean_env_rewards = self.game_env_rewards.get_mean()
                            mean_true_lengths = self.game_true_lengths.get_mean()
                            self.writer.add_scalar('sea/achv_reward/step', mean_achv_rewards[0], frame)
                            self.writer.add_scalar('sea/achv_reward/iter', mean_achv_rewards[0], epoch_num)
                            self.writer.add_scalar('sea/achv_reward/time', mean_achv_rewards[0], total_time)
                            self.writer.add_scalar('sea/env_reward/step', mean_env_rewards[0], frame)
                            self.writer.add_scalar('sea/env_reward/iter', mean_env_rewards[0], epoch_num)
                            self.writer.add_scalar('sea/env_reward/time', mean_env_rewards[0], total_time)
                            self.writer.add_scalar('sea/true_episode_lengths/step', mean_true_lengths, frame)
                            self.writer.add_scalar('sea/true_episode_lengths/iter', mean_true_lengths, epoch_num)
                            self.writer.add_scalar('sea/true_episode_lengths/time', mean_true_lengths, total_time)
                            self.mean_rewards = mean_achv_rewards[0]
                            for key, meter in self.achievement_completions.items():
                                self.writer.add_scalar(f'achievement/{key}', meter.get_mean(), frame)
                            for key, meter in self.objective_rewards.items():
                                self.writer.add_scalar(f'objective_reward/{key}', meter.get_mean(), frame)
                    else:
                        self.writer.add_scalar('sea/env_reward/step', mean_rewards[0], frame)
                        self.writer.add_scalar('sea/env_reward/iter', mean_rewards[0], epoch_num)
                        self.writer.add_scalar('sea/env_reward/time', mean_rewards[0], total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if self.use_sea:
                        torch_ext.save_checkpoint(os.path.join(self.nn_dir, 'objective_selector'), self.vec_env.get_selector_states())

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()

            if should_exit:
                if self.use_sea:
                    if self.save_achievement_buffer:
                        self.achievement_buffer.save(os.path.join(self.nn_dir, 'achievement_buffer'))
                    torch_ext.save_checkpoint(os.path.join(self.nn_dir, 'objective_selector'), self.vec_env.get_selector_states())
                    

            if should_exit:
                return self.last_mean_rewards, epoch_num


class ContinuousA2CBase(A2CBase):

    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)

        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)
        self.rnd_loss_coef = self.config.get('rnd_loss_coef', 1.0)
        self.rnd_reward_coef = self.config.get('rnd_reward_coef', 1.0)

        self.clip_actions = self.config.get('clip_actions', True)

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()
        
        if self.use_sea:
            self.train_sea()

        a_losses = []
        c_losses = []
        b_losses = []
        rnd_losses = []
        entropies = []
        kls = []

        if not self.no_train_actor_critic:
            for mini_ep in range(0, self.mini_epochs_num):
                ep_kls = []
                for i in range(len(self.dataset)):
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, rnd_loss = self.train_actor_critic(self.dataset[i])
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    rnd_losses.append(rnd_loss)
                    ep_kls.append(kl)
                    entropies.append(entropy)
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)

                    self.dataset.update_mu_sigma(cmu, csigma)
                    if self.schedule_type == 'legacy':
                        av_kls = kl
                        if self.multi_gpu:
                            dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                            av_kls /= self.rank_size
                        self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                        self.update_lr(self.last_lr)

                av_kls = torch_ext.mean_list(ep_kls)
                if self.multi_gpu:
                    dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                    av_kls /= self.rank_size
                if self.schedule_type == 'standard':
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

                kls.append(av_kls)
                self.diagnostics.mini_epoch(self, mini_ep)
                if self.normalize_input:
                    self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch
        else:
            last_lr = 0
            lr_mul = 0

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, rnd_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        objectives = batch_dict['obses'].get('objective', None)

        advantages = returns - values

        if self.normalize_value:
            # self.value_mean_std.train()
            # values = self.value_mean_std(values)
            # returns = self.value_mean_std(returns)
            # self.value_mean_std.eval()
            values = self.model.norm_value(values, objectives)
            returns = self.model.norm_value(returns, objectives)

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    if objectives is not None:
                        for i in range(objectives.shape[1]):
                            indices = objectives[:, i].nonzero()
                            if indices.shape[0] > 0:
                                adv = advantages[indices]
                                advantages[indices] = (adv - adv.mean()) / (adv.std() + 1e-8)
                    else:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, rnd_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            self.vec_env.objective_selector.print()

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            # reset_ids = []
            # # for achv, meter in self.achievement_completions.items():
            # #     achv_id = self.vec_env.objective_selector.objective_map[achv]
            # #     completion_rate = meter.get_mean().mean()
            # #     if completion_rate < 1e-2 and meter.current_size >= meter.max_size:
            # #         print(f"{achv} too low completion rate ({completion_rate}), resetting net.")
            # #         reset_ids.append(achv_id)
            # #         meter.clear()

            # if 'explore-0' in self.objective_rewards:
            #     meter = self.objective_rewards['explore-0']
            #     explore_rewards = meter.get_mean().mean()
            #     print(explore_rewards, meter.current_size, meter.max_size)
            #     if explore_rewards < 1e-3 and meter.current_size >= meter.max_size:
            #         print(f"explore reward too low ({explore_rewards}), resetting net.")
            #         reset_ids.append(0)
            #         meter.clear()
            # # else:
            # #     print(self.objective_rewards.keys())

            # for achv_id in reset_ids:
            #     self.model.a2c_network.nets[achv_id].reset()
            #     self.optimizers[achv_id] = self.optimizer_fns[achv_id]()

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, rnd_losses, entropies, kls, last_lr, lr_mul, frame,
                                scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    # self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)
                    write_dict_stats(self.writer, 'losses/bounds_loss', get_dict_losses(b_losses), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.use_sea:
                        if self.game_achv_rewards.current_size > 0:
                            mean_achv_rewards = self.game_achv_rewards.get_mean()
                            mean_env_rewards = self.game_env_rewards.get_mean()
                            mean_true_lengths = self.game_true_lengths.get_mean()
                            self.writer.add_scalar('sea/achv_reward/step', mean_achv_rewards[0], frame)
                            self.writer.add_scalar('sea/achv_reward/iter', mean_achv_rewards[0], epoch_num)
                            self.writer.add_scalar('sea/achv_reward/time', mean_achv_rewards[0], total_time)
                            self.writer.add_scalar('sea/env_reward/step', mean_env_rewards[0], frame)
                            self.writer.add_scalar('sea/env_reward/iter', mean_env_rewards[0], epoch_num)
                            self.writer.add_scalar('sea/env_reward/time', mean_env_rewards[0], total_time)
                            self.writer.add_scalar('sea/true_episode_lengths/step', mean_true_lengths, frame)
                            self.writer.add_scalar('sea/true_episode_lengths/iter', mean_true_lengths, epoch_num)
                            self.writer.add_scalar('sea/true_episode_lengths/time', mean_true_lengths, total_time)
                            self.mean_rewards = mean_achv_rewards[0]
                            for key, meter in self.achievement_completions.items():
                                self.writer.add_scalar(f'achievement/{key}', meter.get_mean(), frame)
                            for key, meter in self.objective_rewards.items():
                                self.writer.add_scalar(f'objective_reward/{key}', meter.get_mean(), frame)
                    else:
                        self.writer.add_scalar('sea/env_reward/step', mean_rewards[0], frame)
                        self.writer.add_scalar('sea/env_reward/iter', mean_rewards[0], epoch_num)
                        self.writer.add_scalar('sea/env_reward/time', mean_rewards[0], total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if self.use_sea:
                        torch_ext.save_checkpoint(os.path.join(self.nn_dir, 'objective_selector'), self.vec_env.get_selector_states())

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()

            if should_exit:
                if self.use_sea:
                    if self.save_achievement_buffer:
                        self.achievement_buffer.save(os.path.join(self.nn_dir, 'achievement_buffer'))
                    torch_ext.save_checkpoint(os.path.join(self.nn_dir, 'objective_selector'), self.vec_env.objective_selector.export())

            if should_exit:
                return self.last_mean_rewards, epoch_num
