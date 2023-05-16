import os
import torch
from torch import nn
import torch.distributed as dist
import gym
import numpy as np
import copy
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.common  import common_losses
from rl_games.common import datasets
from rl_games.common import schedulers


def flatten01(arr):
    s = arr.size()
    return arr.reshape(s[0] * s[1], *s[2:])

def unflatten01(arr, dim0):
    s = arr.size()
    return arr.reshape(dim0, s[0] // dim0, *s[1:])


class SEATrain(nn.Module):
    def __init__(self, state_shape, value_size, ppo_device, num_agents, horizon_length, num_actors, num_actions, 
                seq_len, normalize_value, network, config, writter, max_epochs, multi_gpu):
        nn.Module.__init__(self)

        self.ppo_device = ppo_device
        self.num_agents, self.horizon_length, self.num_actors, self.seq_len = num_agents, horizon_length, num_actors, seq_len
        self.normalize_value = normalize_value
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.value_size = value_size
        self.max_epochs = max_epochs
        self.multi_gpu = multi_gpu
        self.truncate_grads = config.get('truncate_grads', False)
        self.config = config
        # self.normalize_input = config['normalize_input']
        self.normalize_input = False

        state_config = {
            'value_size' : value_size,
            'input_shape' : state_shape,
            'actions_num' : num_actions,
            'num_agents' : num_agents,
            'num_seqs' : num_actors,
            'normalize_input' : self.normalize_input,
            'normalize_value': self.normalize_value,
            'use_sea' : True
        }

        self.model = network.build(state_config)
        self.lr = float(config['learning_rate'])
        self.linear_lr = config.get('lr_schedule') == 'linear'

        # todo: support max frames as well
        if self.linear_lr:
            self.scheduler = schedulers.LinearScheduler(self.lr, 
                max_steps = self.max_epochs, 
                apply_to_entropy = False,
                start_entropy_coef = 0)
        else:
            self.scheduler = schedulers.IdentityScheduler()
        
        self.mini_epoch = config['mini_epochs']
        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))

        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)

        self.writter = writter
        self.weight_decay = config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), float(self.lr), eps=1e-08, weight_decay=self.weight_decay)
        self.frame = 0
        self.epoch_num = 0
        self.running_mean_std = None
        self.grad_norm = config.get('grad_norm', 1)
        self.truncate_grads = config.get('truncate_grads', False)
        self.e_clip = config.get('e_clip', 0.2)
        self.truncate_grad = self.config.get('truncate_grads', False)

        self.centroids = None

        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            # dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)

            self.device_name = 'cuda:' + str(self.rank)
            config['device'] = self.device_name
            if self.rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device_name)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_stats_weights(self, model_stats=False):
        state = {}
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights): 
        pass

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

    def forward(self, input_dict):
        return self.model(input_dict)
    
    def get_sea_outputs(self, obs, next_obs, action):
        self.model.eval()
        processed_obs = self._preproc_obs(obs)
        processed_next_obs = self._preproc_obs(next_obs)

        # print(obs.shape, processed_obs.shape)
        # print(next_obs.shape, processed_next_obs.shape)
        # print(action.shape)
        
        with torch.no_grad():
            return self.model({
                'obs': processed_obs,
                'next_obs': processed_next_obs,
                'action': action
            })

    def train_net(self, achievement_buffer, sea_buffer, frame):
        batch_size = self.minibatch_size
        print(achievement_buffer.size, batch_size, self.writter)
        # if achievement_buffer.size >= batch_size:
        if achievement_buffer.size >= batch_size // 4:
            contrast_loss, pred_loss = 0, 0
            self.train()
            for _ in range(self.mini_epoch):
                a_obses, a_actions, a_next_obses, valids = achievement_buffer.sample(batch_size)
                s_obses, s_actions, s_next_obses, rewards = sea_buffer.sample(batch_size * a_obses.shape[1])

                # v = rewards.shape[0]
                # for i in range(v):
                #     if rewards[i] > 0.1:
                #         print(s_obses[i, 0], s_next_obses[i, 0], rewards[i])

                _contrast_loss, _pred_loss = self.calc_gradients({
                    'obs': a_obses,
                    'action': a_actions,
                    'next_obs': a_next_obses,
                    'valid': valids
                }, {
                    'obs': s_obses,
                    'action': s_actions,
                    'next_obs': s_next_obses,
                    'reward': rewards
                })
                contrast_loss += _contrast_loss
                pred_loss += _pred_loss

                if self.normalize_input:
                    self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

            avg_contrast_loss = contrast_loss / self.mini_epoch
            avg_pred_loss = pred_loss / self.mini_epoch

            self.epoch_num += 1
            self.lr, _ = self.scheduler.update(self.lr, 0, self.epoch_num, 0, 0)
            self.update_lr(self.lr)
            if self.writter != None:
                self.writter.add_scalar('losses/sea_contrast_loss', avg_contrast_loss, frame)
                self.writter.add_scalar('losses/sea_pred_loss', avg_pred_loss, frame)
                self.writter.add_scalar('info/sea_lr', self.lr, frame)
            return avg_contrast_loss, avg_pred_loss
        else:
            return 0, 0
        
    def _calc_contrast_loss(self, batch):
        obses, actions, next_obses, valids = batch['obs'], batch['action'], batch['next_obs'], batch['valid']
        B, L = obses.shape[:2]
        valid_masks = valids.type(dtype=torch.float32)
        obses, actions, next_obses, valid_masks = map(flatten01, (obses, actions, next_obses, valid_masks))

        processed_obs = self._preproc_obs(obses)
        processed_next_obs = self._preproc_obs(next_obses)
        
        embeds, _ = self.model({
            'obs': processed_obs,
            'next_obs': processed_next_obs,
            'action': actions
        })

        if self.centroids is None:
            self.centroids = torch.randn((8, embeds.shape[-1]), device=embeds.device, requires_grad=False)

        centroids = self.centroids.clone()

        # print(processed_obs[:10, 0])
        # print(processed_next_obs[:10, 0])

        embeds = embeds * valid_masks.unsqueeze(-1)
        flat_embeds = embeds
        embeds = unflatten01(embeds, B)
        valid_masks = unflatten01(valid_masks, B)
        mat_valid_masks = valid_masks.unsqueeze(1) * valid_masks.unsqueeze(2)
        # embed_dis = torch.square(torch.cdist(embeds, embeds))  # [B, L, L]
        embed_dis = torch.square(embeds.unsqueeze(1) - embeds.unsqueeze(2)).sum(-1)  # [B, L, L]
        # print(torch.square(embed_dis - embed_dis_2).sum())
        embed_dis = embed_dis * mat_valid_masks
        # kernel_dis = torch.exp(-embed_dis / (embed_dis.max() + 1e-6) * 4)
        kernel_dis = torch.exp(-embed_dis / (embed_dis.max()) * 2)
        kernel_dis = kernel_dis * mat_valid_masks + torch.eye(L, device=kernel_dis.device).unsqueeze(0) * (1 - mat_valid_masks)
        # print(kernel_dis[0])
        sea_losses = -torch.det(kernel_dis)
        loss = sea_losses.mean()

        # C = centroids.shape[0]
        # cent_dis = torch.cdist(flat_embeds.unsqueeze(0), centroids.unsqueeze(0))[0]  # [B*L, C]
        # belong_embeds = [[] for _ in range(C)]
        # cent_loss = 0
        # for i in range(B * L):
        #     c = cent_dis[i].argmin()
        #     belong_embeds[c].append(flat_embeds[i].detach())
        #     cent_loss += cent_dis[i][c]
        # for i in range(C):
        #     if len(belong_embeds[i]) > 0:
        #         self.centroids[i] = torch.stack(belong_embeds[i], 0).mean(0).detach()
        
        # cent_loss /= B * L

        # print(sea_losses.mean().item(), cent_loss.item())

        # loss += cent_loss * 10

        # S = int(np.sqrt(B))
        # indices = np.random.randint(B, size=S)
        # # print(indices)
        # # print(valids[indices])
        # # seleceted_embeds = embeds[indices]  # [S, L, embeds]
        # dis_sum = 0
        # dis_tot = 0
        # INF = 1e9
        # for i in range(S):
        #     min_dis = torch.ones((L,), device=embeds.device) * INF
        #     for j in range(S):
        #         if i != j:
        #             dis = torch.square(torch.cdist(embeds[indices[i]], embeds[indices[j]]))  # [L, L]
        #             # print(dis)
        #             mat_valid_mask = valid_masks[indices[i]].unsqueeze(1) * valid_masks[indices[j]].unsqueeze(0)
        #             dis = dis * mat_valid_mask + INF * (1 - mat_valid_mask)
        #             min_dis = torch.minimum(min_dis, torch.min(dis, 1).values)
        #     # print(min_dis)
        #     dis_sum += (min_dis * valid_masks[indices[i]]).sum()
        #     dis_tot += valid_masks[indices[i]].sum()

        # # print(dis_sum, dis_tot)

        #         # for xi in range(L):
        #         #     if valid_masks[indices[i]][xi]:
        #         #         for xj in range(L):
        #         #             if valid_masks[indices[j]][xj]:
        #         #                 dis_tot += 1
        #         #                 dis_sum += torch.square(embeds[indices[i]][xi] - embeds[indices[j]][xj]).sum()
        # if dis_tot > 0:
        #     loss += dis_sum / dis_tot * 0.2

        return loss
    
    def _calc_pred_loss(self, batch):
        obses, actions, next_obses, rewards = batch['obs'], batch['action'], batch['next_obs'], batch['reward']

        processed_obs = self._preproc_obs(obses)
        processed_next_obs = self._preproc_obs(next_obses)
        
        _, preds = self.model({
            'obs': processed_obs,
            'next_obs': processed_next_obs,
            'action': actions
        })

        v = rewards.shape[0]
        ind = np.random.randint(v, size=1)
        # for i in ind:
        #     print(processed_obs[i, 0].item(), processed_next_obs[i, 0].item(), rewards[i].item(), preds[i].item())

        # loss = 0.5 * torch.square(rewards - preds).mean()
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(preds[:, 0], rewards)

        return loss


    def calc_gradients(self, contrast_batch, pred_batch):
        contrast_loss = self._calc_contrast_loss(contrast_batch)
        pred_loss = self._calc_pred_loss(pred_batch)
        loss = contrast_loss + pred_loss * 0
        
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        loss.backward()

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
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.optimizer.step()
        
        return contrast_loss, pred_loss
