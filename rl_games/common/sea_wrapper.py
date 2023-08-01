import numpy as np
import gym
from rl_games.algos_torch import torch_ext
import torch
import copy
import bisect

from rl_games.common.ivecenv import IVecEnv, IVecEnvWrapper


MAX_OBJECTIVES = 50
START = 0.1
STEP = 1.5
# TARGET_DISTANCES = [0.1 * i for i in range(20)] + [2 * 1.2 ** i for i in range(30)]
TARGET_DISTANCES = [1 * 1.5 ** i for i in range(50)]
# TARGET_DISTANCES = [i + 1 for i in range(50)]


def enlarge(arr, diff, start=-1):
    # diff = n - arr.shape[0]
    if diff > 0:
        n = arr.shape[0]
        arr = np.pad(arr, [[0, diff]] * len(arr.shape), constant_values=0)
        if start != -1:
            pos = start if start >= 0 else start + n
            for i in range(len(arr.shape)):
                arr = np.concatenate((np.take(arr, range(pos + 1), i), np.take(arr, range(n, n + diff, i)), np.take(arr, range(pos + 1, n))), i)
    return arr

def remove(arr, n):  # remove id 1 to 1+n-1
    for i in range(len(arr.shape)):
        arr = np.delete(arr, range(1, 1 + n), i)
    return arr


class WindowedCounter:
    def __init__(self, window_size, num_objects):
        self.window_size = window_size
        self.num_objects = num_objects
        self.counter = np.zeros((window_size, num_objects), dtype=int)
        self.size = 0

    def update_num_objects(self, num_objects):
        new_counter = np.zeros((self.window_size, num_objects), dtype=int)
        new_counter[:, :self.num_objects] = self.counter.copy()
        self.counter = new_counter
        self.num_objects = num_objects

    def update(self, counts):
        counts = np.array(counts)
        assert counts.shape[0] == self.num_objects
        self.counter[self.size % self.window_size] = counts.copy()
        self.size += 1

    def get_freq(self):
        return self.counter.sum(0) / max(1, min(self.size, self.window_size))


class ObjectiveSelector:
    def __init__(self, no_explore=False, num_exploration_objectives=1):
        self.no_explore = no_explore
        self.num_exploration_policies = num_exploration_objectives
        self.known_objectives = [f'explore-{i}' for i in range(self.num_exploration_policies)] #+ ['reach_cabinet']
        self.num_objectives = len(self.known_objectives)
        n = self.num_objectives
        self.objective_map = {c: i for i, c in enumerate(self.known_objectives)}
        self.np_random = np.random.RandomState()
        self.objectives = None
        self.windowed_completion_counter = WindowedCounter(1000, n)
        self.completion_counter = np.zeros((n,), dtype=int)
        self.counter = np.zeros((n, n), dtype=int)
        self.direct_counter = np.zeros((n, n), dtype=int)
        self.objective_counter = np.zeros((n, n), dtype=int)
        self.objective_total = np.zeros((n,), dtype=int)
        self.total_games = 0
        self.original_completion_counter = np.zeros((n,), dtype=int)
        self.original_total_games = 0
        self.original_counter = np.zeros((n, n), dtype=int)
        self.original_direct_counter = np.zeros((n, n), dtype=int)
        self.batch_reward = None
        self.objective_reward = [None] * n

    def seed(self, seed):
        self.np_random.seed(seed)

    def reset(self, batch_size, obs):
        self.batch_size = batch_size
        self.completed_objectives = [[] for _ in range(batch_size)]
        self.objectives = np.zeros((batch_size,), dtype=int)
        for i in range(batch_size):
            self.objectives[i] = self.get_init_objective(i)
            # print(i, "init objective:", self.known_objectives[self.objectives[i]])
        self.batch_reward = np.zeros((batch_size,), dtype=np.float)
        return self.objectives
    
    def get_step_completed(self, step_completed):
        ret_step_completed = [[] for _ in range(self.batch_size)]
        new_objectives = 0
        for i in range(self.batch_size):
            for c in step_completed[i]:
                if c not in self.completed_objectives[i]:
                    if c not in self.known_objectives:
                        self.known_objectives.append(c)
                        self.objective_map[c] = self.num_objectives
                        self.completion_counter = enlarge(self.completion_counter, 1)
                        self.objective_reward.append(None)
                        self.num_objectives += 1
                        new_objectives += 1
                    self.completed_objectives[i].append(c)
                    ret_step_completed[i].append(c)
                    self.completion_counter[self.objective_map[c]] += 1
        if new_objectives > 0:
            self.objective_counter = enlarge(self.objective_counter, new_objectives)
            self.counter = enlarge(self.counter, new_objectives)
            self.direct_counter = enlarge(self.direct_counter, new_objectives)
            self.objective_total = enlarge(self.objective_total, new_objectives)
            self.windowed_completion_counter.update_num_objects(self.num_objectives)
            self.update_known_objectives()
        return ret_step_completed, step_completed

    def step(self, step_completed, step_completed_full, obs, reward, is_done, info):
        if isinstance(is_done, np.ndarray):
            obj_done = np.zeros(is_done.shape, dtype=np.bool)
            obj_rew = np.zeros(is_done.shape, dtype=float)
        else:
            obj_done = torch.zeros(is_done.shape, dtype=torch.bool, device=is_done.device)
            obj_rew = torch.zeros(is_done.shape, dtype=torch.float32, device=is_done.device)
        for i in range(self.batch_size):
            original_objective = self.objectives[i]
            if len(step_completed[i]) > 0:
                cidx = [0] + [self.objective_map[c] for c in self.completed_objectives[i]]
                for j in range(len(cidx) - len(step_completed[i]), len(cidx)):
                    # print(j, len(cidx))
                    cj = cidx[j]
                    self.objective_counter[original_objective][cj] += 1
                    self.objective_total[original_objective] += 1
                    self.direct_counter[cidx[j - 1]][cj] += 1
                    for k in range(j):
                        self.counter[cidx[k]][cj] += 1
            self.objectives[i], obj_rew[i], obj_done[i] = self.get_next_objective(i, obs[i], reward[i], is_done[i], info, self.completed_objectives[i], self.objectives[i], step_completed[i], step_completed_full[i])
            # if len(step_completed[i]) > 0:
            #     print(i, "completed:", self.completed_objectives[i], "step_completed:", step_completed[i], "original objective:", self.known_objectives[original_objective], "new objective:", self.known_objectives[self.objectives[i]])
            self.batch_reward[i] += obj_rew[i]
            if obj_done[i] or is_done[i]:
                rew = self.batch_reward[i]
                if self.objective_reward[original_objective] is None:
                    self.objective_reward[original_objective] = rew
                else:
                    self.objective_reward[original_objective] = self.objective_reward[original_objective] * 0.99 + rew * 0.01
                if len(step_completed[i]) == 0:
                    self.objective_total[original_objective] += 1
                self.batch_reward[i] = 0.
            if is_done[i]:
                counts = np.zeros((self.num_objectives,), dtype=int)
                for c in self.completed_objectives[i]:
                    counts[self.objective_map[c]] = 1
                self.windowed_completion_counter.update(counts)
                self.completed_objectives[i] = []
                obj_done[i] = True
                self.total_games += 1
                self.objectives[i] = self.get_init_objective(i)
                # print(i, "init objective:", self.known_objectives[self.objectives[i]])
        return self.objectives, obj_rew, obj_done
    
    def update_known_objectives(self):
        pass

    def get_init_objective(self, i_batch):
        pass
    
    def get_next_objective(self, i_batch, obs, reward, done, info, completed, objective, step_completed, step_completed_full):
        pass

    def print(self):
        print("Known achievements:")
        for i, achv in enumerate(self.known_objectives):
            print(f"{i}:{achv}")

        print("")
        print("Objective total")
        print(self.objective_total)
        print("objective counter")
        print(self.objective_counter)
        print(self.objective_counter / (np.array(self.objective_total)[:, None] + 1))

        print("")
        print("Windowed completion counter:")
        print(self.windowed_completion_counter.get_freq())
        
        print("")
        print("Completion counter:")
        counter = np.array(self.completion_counter)
        counter[:len(self.original_completion_counter)] -= np.array(self.original_completion_counter)
        print(counter)
        print(counter / (self.total_games - self.original_total_games))

        print("")
        print("Objective reward:")
        print(self.objective_reward)

        n = self.num_objectives
        on = self.original_counter.shape[0]

        print("")
        print("Direct counter:")
        dc = self.direct_counter.copy()
        dc[:on, :on] -= self.original_direct_counter
        for i in range(n):
            for j in range(n):
                print(dc[i, j], end=',' if j < n - 1 else '\n')

        print("")
        print("Counter:")
        c = self.counter.copy()
        c[:on, :on] -= self.original_counter
        for i in range(n):
            for j in range(n):
                print(c[i, j], end=',' if j < n - 1 else '\n')

    def export(self):
        diff = self.num_exploration_policies - 1
        return {
            'num_objectives': self.num_objectives - self.num_exploration_policies,
            'known_objectives': self.known_objectives[self.num_exploration_policies:],
            # 'objective_map': self.objective_map,
            'total_games': self.total_games,
            'completion_counter': remove(self.completion_counter, diff),
            'counter': remove(self.counter, diff),
            'direct_counter': remove(self.direct_counter, diff),
            # 'objective_counter': remove(self.objective_counter, self.num_exploration_policies - 1),
            # 'objective_total': remove(self.objective_total, self.num_exploration_policies - 1),
        }
    
    def load(self, states):
        num_objectives = states['num_objectives']
        self.num_objectives = n = num_objectives + self.num_exploration_policies
        diff = self.num_exploration_policies - 1
        self.known_objectives = [f'explore-{i}' for i in range(self.num_exploration_policies)] + states['known_objectives']
        # self.objective_map = states['objective_map']
        self.objective_map = {k: i for i, k in enumerate(self.known_objectives)}
        self.total_games = states['total_games']
        self.completion_counter = enlarge(states['completion_counter'], diff, 0)
        self.counter = enlarge(states['counter'], diff, 0)
        self.direct_counter = enlarge(states['direct_counter'], diff, 0)
        # self.objective_counter = states['objective_counter']
        # self.objective_total = states['objective_total']

        self.objective_counter = np.zeros((n, n), dtype=int)
        self.objective_total = np.zeros((n,), dtype=int)
        self.windowed_completion_counter = WindowedCounter(1000, self.num_objectives)

        self.original_completion_counter = copy.deepcopy(self.completion_counter)
        self.original_total_games = self.total_games
        self.objective_reward = [None] * self.num_objectives
        self.original_counter = self.counter.copy()
        self.original_direct_counter = self.direct_counter.copy()


class RandomObjectiveSelector(ObjectiveSelector):
    def set_num_objectives(self, num_objectives):
        self._num_objectives = num_objectives

    def _get_random_objective(self):
        if self.no_explore:
            return self.np_random.randint(1, self._num_objectives)
        else:
            return self.np_random.randint(self._num_objectives)
    
    def get_init_objective(self, i_batch):
        return self._get_random_objective()
    
    def get_next_objective(self, i_batch, obs, reward, done, info, completed, objective, step_completed, step_completed_full):
        return self._get_random_objective(), len(step_completed), False
    

REWARDS_SCALING = {
    # "robot_close_switch_0.25": 1,
    # "switch_pulled": 2,
    # "robot_close_cabinet_0.2": 10,
    # # "gripper_around_drawer_handle": 15,
    # "open_cabinet_0.01": 40,
    # "open_cabinet_0.2": 40,
    # "open_cabinet_0.39": 80
}

REWARDS_PREREQUISITES = {
    # "robot_close_switch_0.25": [],
    # "switch_pulled": [],
    # "robot_close_cabinet_0.2": [],
    # "gripper_around_drawer_handle": [],
    # "open_cabinet_0.1": ["gripper_around_drawer_handle"],
    # "open_cabinet_0.39": ["gripper_around_drawer_handle"]
}

REWARDS_UNTILS = {
    # "robot_close_switch_0.25": ["switch_pulled"],
    # "switch_pulled": [],
    # "robot_close_cabinet_0.2": [],
    # "gripper_around_drawer_handle": [],
    # "open_cabinet_0.1": [],
    # "open_cabinet_0.39": []
}


class DummyObjectiveSelector(ObjectiveSelector):
    def set_num_objectives(self, num_objectives):
        self._num_objectives = num_objectives
    
    def get_init_objective(self, i_batch):
        return self.np_random.randint(self.num_exploration_policies)
    
    def get_next_objective(self, i_batch, obs, reward, done, info, completed, objective, step_completed, step_completed_full):
        # reward = sum([REWARDS_SCALING.get(c, 1) if all([r in completed for r in REWARDS_PREREQUISITES.get(c, [])]) else 0 for c in step_completed])
        # print(step_completed_full)
        # reward = sum([REWARDS_SCALING.get(c, 1) 
        #               if all([r in completed for r in REWARDS_PREREQUISITES.get(c, [])]) and all([r not in completed for r in REWARDS_UNTILS.get(c, [])]) else 0 for c in step_completed_full])
        reward = len(step_completed)
        return objective, reward, False
    

class DummyDenseObjectiveSelector(DummyObjectiveSelector):   
    def get_next_objective(self, i_batch, obs, reward, done, info, completed, objective, step_completed, step_completed_full):
        return objective, reward, False
    

class DummySequenceObjectiveSelector(ObjectiveSelector):
    def __init__(self, sequence, **kwargs):
        super().__init__(**kwargs)
        self.sequence = sequence

    def set_num_objectives(self, num_objectives):
        self._num_objectives = num_objectives
    
    def get_init_objective(self, i_batch):
        return self.objective_map[self.sequence[0]]
    
    def get_next_objective(self, i_batch, obs, reward, done, info, completed, objective, step_completed, step_completed_full):
        # reward = sum([REWARDS_SCALING.get(c, 1) if all([r in completed for r in REWARDS_PREREQUISITES.get(c, [])]) else 0 for c in step_completed])
        # print(step_completed_full)
        # reward = sum([REWARDS_SCALING.get(c, 1) 
        #               if all([r in completed for r in REWARDS_PREREQUISITES.get(c, [])]) and all([r not in completed for r in REWARDS_UNTILS.get(c, [])]) else 0 for c in step_completed_full])
        if len(step_completed) > 0 and objective >= self.num_exploration_policies:
            reward = 0
            before_completed = completed[:-len(step_completed)]
            pos = 0
            while pos < len(self.sequence) and self.sequence[pos] in before_completed:
                pos += 1
            for c in step_completed:
                if pos == len(self.sequence):
                    break
                if c == self.sequence[pos]:
                    reward += 1
                before_completed.append(c)
                while pos < len(self.sequence) and self.sequence[pos] in before_completed:
                    pos += 1
            min_acc = min([self.completion_counter[self.objective_map[c]] if c in self.objective_map else 0 for c in self.sequence])
            ratio = min(min_acc / 200, 1)
            reward = reward + (len(step_completed) - reward) * (max(ratio * 0.5, 0.0) + 0.1)
            if pos < len(self.sequence):
                next_obj = self.sequence[pos]
                print("next objective:", next_obj)
                objective = self.objective_map[next_obj]
            else:
                print("next objective:", self.known_objectives[0])
                objective = 0
        else:
            reward = 0
        return objective, reward, False    


class SequentialObjectiveSelector(ObjectiveSelector):
    def fill_objectives(self, num_objectives):
        for i in range(self.num_objectives, num_objectives + 1):
            self.known_objectives.append(i - 1)
            self.objective_map[i - 1] = i
        self.num_objectives = num_objectives

    def get_init_objective(self, i_batch):
        return 1 if self.num_objectives > 1 else 0
    
    def get_next_objective(self, i_batch, obs, done, info, completed, objective, step_completed, step_completed_full):
        if len(step_completed) > 0:
            return min(max([self.objective_map[c] for c in step_completed]) + 1, self.num_objectives - 1), len(step_completed), False
        else:
            return objective, 0, False
        

# class EmpiricalObjectiveSelector(ObjectiveSelector):
#     def __init__(self):
#         super().__init__()
#         self.counter = None
#         self.direct_counter = None

#     def reset(self, batch_size):
#         super().reset(batch_size)

#     def update_known_objectives(self):
#         n = self.num_objectives
#         new_counter = np.zeros((n, n), dtype=int)
#         new_direct_counter = np.zeros((n, n), dtype=int)
#         if self.counter is not None:
#             _n = self.counter.shape[0]
#             new_counter[:_n, :_n] = self.counter
#             new_direct_counter[:_n, :_n] = self.direct_counter
#         self.counter = new_counter
#         self.direct_counter = new_direct_counter

#     def _get_next_objective(self, last_completed):
#         if self.direct_counter is None:
#             return 0
#         else:
#             selection = np.argmax(self.direct_counter[last_completed])
#             if self.direct_counter[last_completed, selection] == 0:
#                 selection = 0
#             # print('next', last_completed, selection)
#             return selection

#     def get_next_objective(self, completed, objective, step_completed):
#         if len(step_completed) > 0:
#             cidx = [0] + [self.objective_map[completed[i]] for i in range(len(completed))]
#             for j in range(len(completed) - len(step_completed) + 1, len(cidx)):
#                 self.direct_counter[cidx[j - 1], cidx[j]] += 1
#                 for i in range(j):
#                     self.counter[cidx[i], cidx[j]] += 1
#             next_obj = self._get_next_objective(self.objective_map[step_completed[-1]])
#             obj_done = False
#             if objective == 0:
#                 obj_reward = len(step_completed)
#             else:
#                 if self.known_objectives[objective] in step_completed:
#                     obj_reward = 1.
#                     obj_done = True
#                 else:
#                     obj_reward = 0.
#             return next_obj, obj_reward, obj_done
#         else:
#             return objective, 0., False
        
#     def get_init_objective(self):
#         return self._get_next_objective(0)


def find_earliest(graph, u):
    n = graph.shape[0]
    nl = [[u], []]
    all_nodes = set([u])
    d = 0
    cnt = 0
    while len(nl[d]) > 0:
        nl[1 - d] = []
        for v in nl[d]:
            for x in range(n):
                if graph[x][v] and not x in all_nodes:
                    nl[1 - d].append(x)
                    all_nodes.add(x)
                    cnt += 1
        d = 1 - d
    return cnt


class NewEmpiricalObjectiveSelector(ObjectiveSelector):
    def __init__(self, no_explore=False, num_exploration_objectives=1, exploration_only=False, follow_traj_prob=0.8):
        super().__init__(no_explore, num_exploration_objectives)
        n = self.num_objectives
        self.correct_counter = WindowedCounter(100, n)
        self.exploration_only = exploration_only
        self.original_counter = self.counter.copy()
        self.original_direct_counter = self.direct_counter.copy()
        self.trajectories = [[] for _ in range(n)]
        self.state_trajectories = [[] for _ in range(n)]
        self.cur_trajectory = None
        self.last_updated_dep_graph = None
        self.follow_traj_prob = follow_traj_prob
        self.intrinsic_reward = None
        self.exploring = None

    def export(self):
        to_export = super().export()
        return {
            'counter': self.counter,
            'direct_counter': self.direct_counter,
            'state_trajectories': self.state_trajectories,
            **to_export
        }
    
    def load(self, states):
        # self.counter = states.pop('counter')
        # self.direct_counter = states.pop('direct_counter')
        super().load(states)
        self.original_known_objectives = [obj for i, obj in enumerate(self.known_objectives) if self.completion_counter[i] / self.total_games > 0.01]
        print("original known objectievs:", self.original_known_objectives)
        # self.original_counter = self.counter.copy()
        # self.original_direct_counter = self.direct_counter.copy()
        self.trajectories = [[] for _ in range(self.num_objectives)]
        self.state_trajectories = [[] for _ in range(self.num_objectives)]

    def reset(self, batch_size, obs):
        self.cur_trajectory = [[] for _ in range(batch_size)]
        self.cur_state_trajectory = [[obs[i].detach().cpu().numpy().copy()] for i in range(batch_size)]
        self.intrinsic_reward = [None for _ in range(batch_size)]
        self.exploring = np.zeros((batch_size,), dtype=int)
        return super().reset(batch_size, obs)

    def update_known_objectives(self):
        n = self.num_objectives
        # new_counter = np.zeros((n, n), dtype=int)
        # new_direct_counter = np.zeros((n, n), dtype=int)
        # new_dep_graph = np.zeros((n, n), dtype=int)
        # if self.counter is not None:
        #     _n = self.counter.shape[0]
        #     new_counter[:_n, :_n] = self.counter
        #     new_direct_counter[:_n, :_n] = self.direct_counter
        #     new_dep_graph[:_n, :_n] = self.dep_graph
        # self.counter = new_counter
        # self.direct_counter = new_direct_counter
        # self.dep_graph = new_dep_graph
        diff = self.num_objectives - self.dep_graph.shape[0]
        self.dep_graph = enlarge(self.dep_graph, diff)
        for _ in range(diff):
            self.trajectories.append([])
            self.state_trajectories.append([])

    def _update_dep_graph(self, force_update=False):
        if force_update or (self.last_updated_dep_graph is None or sum(self.completion_counter) - self.last_updated_dep_graph > 100):
            print("updating dependency graph")
            self.last_updated_dep_graph = sum(self.completion_counter)
            counter = self.counter
            completion = np.array(self.completion_counter)
            n = counter.shape[0]
            graph = np.zeros((n, n), dtype=int)

            for i in range(n):
                for j in range(n):
                    graph[i, j] = completion[j] >= 0 and counter[i, j] > completion[j] * 0.99 and counter[j, i] == 0

            self.dep_graph = graph.copy()

            key_edges = []
            for i in range(n):
                for j in range(n):
                    if graph[i][j] == 1:
                        earliest = find_earliest(graph, j)
                        graph[i][j] = 0
                        connected = find_earliest(graph, j) < earliest
                        graph[i][j] = 1
                        if connected:
                            key_edges.append((i, j))
                        # if connected:
                        #     print(i, '->', j)
                    else:
                        connected = False
                    print(f"{1 if connected else 0}", end='\n' if j == n - 1 else ',')
            print(key_edges)

    def _get_intrinsic_reward(self, obs, obj):
        np_obs = obs.detach().cpu().numpy()
        if len(self.state_trajectories[obj]) > 5:
            n = len(self.state_trajectories[obj])
            ids = self.np_random.choice(n, size=min(5, n))
            trajs = [self.state_trajectories[obj][i] for i in ids]
            n = len(trajs)
            reward = 0
            for traj in trajs:
                dis = np.square(np_obs.reshape(1, -1) - traj).sum(-1)  # [len]
                l = traj.shape[0]
                reward += np.exp(-np.arange(l) / 2 - dis / 10).max() / n
                # print(dis.mean())
                # print(dis.shape, traj.shape, np_obs.shape, reward.shape)
            # print(self.known_objectives[obj], reward)
            return reward
        else:
            return 0

    def _get_next_objective(self, completed, last_completed):
        windowed_freq = self.windowed_completion_counter.get_freq()
        # self.follow_trajectory = None
        if last_completed == 0:  # episode start
            self._update_dep_graph()
            self.follow_trajectory = None
        #     return self.np_random.randint(self.num_objectives)
        #     choices = []
        #     for i in range(self.num_exploration_policies, self.num_objectives):
        #         if len(self.trajectories[i]) > 0 and self.objective_counter[i][i] < 0.1 * (self.objective_total[i] + 1):
        #         # if len(self.trajectories[i]) > 0 and windowed_freq[i] < 0.1:
        #         # if len(self.trajectories[i]) > 0 and self.completion_counter[i] / (self.total_games + 1) < 0.1:
        #             choices.append(i)
        #     if len(choices) > 0 and self.np_random.rand() < self.follow_traj_prob:
        #         c = self.np_random.choice(choices)
        #         trajs = self.trajectories[c]
        #         self.follow_trajectory = trajs[self.np_random.randint(len(trajs))]
        #         print("Following trajectory for", self.known_objectives[c], ":", self.follow_trajectory)
        #         return self.follow_trajectory[0]
        #     else:
        #         self.follow_trajectory = None
        if self.direct_counter is None:
            return self.np_random.randint(self.num_exploration_policies)
        else:
            if self.follow_trajectory is not None:
                for j in self.follow_trajectory:
                    if self.known_objectives[j] not in completed:
                        return j
                return self.np_random.randint(self.num_exploration_policies)
            if self.np_random.rand() < 0.3 and not self.no_explore:
                return self.np_random.randint(self.num_exploration_policies)
            else:
                # known_objectives = self.original_known_objectives if self.exploration_only else self.known_objectives
                known_objectives = self.known_objectives
                remaining = [c for c in known_objectives[self.num_exploration_policies:] if c not in completed]
                remaining = [self.objective_map[c] for c in remaining]
                if len(remaining) == 0:
                    return self.np_random.randint(self.num_exploration_policies)
                if self.np_random.rand() < 0.2 and not self.no_explore:
                    # score = np.array([1 / self.completion_counter[r] for r in remaining]) + 1e-6
                    score = np.ones_like(remaining)
                    # if self.no_explore:
                    #     score[:self.num_exploration_policies] = 0
                    return self.np_random.choice(remaining, p=score / score.sum())
                # print(remaining, self.num_exploration_policies, self.num_objectives, completed)
                # print([(self.known_objectives[v] in completed) for v in range(self.num_exploration_policies, self.num_objectives) if self.dep_graph[v, 1]])
                # print(all([(self.known_objectives[v] in completed) for v in range(self.num_exploration_policies, self.num_objectives) if self.dep_graph[v, 1]]))
                remaining_possible = [c for c in remaining if all([(self.known_objectives[v] in completed) for v in range(self.num_exploration_policies, self.num_objectives) if self.dep_graph[v, c]])]
                # print(remaining_possible)
                # score = np.array([self.direct_counter[last_completed][r] / self.completion_counter[r] / self.completion_counter[r] for r in remaining]) + 1e-6
                # score = np.array([1 / self.completion_counter[r] for r in remaining_possible]) + 1e-6
                score = np.array([self.direct_counter[last_completed][r] / max(1e-3, self.completion_counter[r]) / max(0.1, windowed_freq[r]) for r in remaining_possible]) + 1e-6
                # if self.no_explore:
                #     score[:self.num_exploration_policies] = 0
                # print(score)
                selection = self.np_random.choice(remaining_possible, p=score / score.sum())
                return selection

    def get_next_objective(self, i_batch, obs, reward, done, info, completed, objective, step_completed, step_completed_full):
        # self.cur_trajectory[i_batch] += [self.objective_map[c] for c in step_completed]
        for c in step_completed:
            oc = self.objective_map[c]
            self.cur_trajectory[i_batch].append(oc)
            if len(self.trajectories[oc]) < 30:
                self.trajectories[oc].append(copy.deepcopy(self.cur_trajectory[i_batch]))
                self.state_trajectories[oc].append(np.stack(self.cur_state_trajectory[i_batch], 0)[::-10].copy())
            else:
                self.trajectories[oc][self.np_random.randint(30)] = copy.deepcopy(self.cur_trajectory[i_batch])
                self.state_trajectories[oc][self.np_random.randint(30)] = np.stack(self.cur_state_trajectory[i_batch], 0)[::-10].copy()
        if done:
            self.cur_trajectory[i_batch] = []
        if done or len(step_completed) > 0:
            self.cur_state_trajectory[i_batch] = []
        self.cur_state_trajectory[i_batch].append(obs.detach().cpu().numpy().copy().reshape(-1))
        obj_reward = 0.
        # obj_reward += info.get("achievement_dense_rewards", {}).get("penalty", 0.)
        if objective >= self.num_exploration_policies and not self.exploring[i_batch]:
            # new_intrinsic = self._get_intrinsic_reward(obs, objective)
            # # diff = new_intrinsic - self.intrinsic_reward[i_batch]
            # obj_reward = new_intrinsic * max(0.5 - self.windowed_completion_counter.get_freq()[objective], 0) * 0.01
            # # print(diff, self.known_objectives[objective])
            # self.intrinsic_reward[i_batch] = new_intrinsic

            if "achievement_dense_rewards" in info:
                achv_id = info["achievement_list"].index(self.known_objectives[objective])
                new_intrinsic = info["achievement_dense_rewards"][i_batch][achv_id]
                obj_reward += new_intrinsic #- self.intrinsic_reward[i_batch]
                self.intrinsic_reward[i_batch] = new_intrinsic
                # print(obj_reward)
            pass
        if len(step_completed) > 0:
            cidx = list(range(self.num_exploration_policies)) + [self.objective_map[completed[i]] for i in range(len(completed))]
            # for j in range(len(completed) - len(step_completed) + self.num_exploration_policies, len(cidx)):
            #     self.direct_counter[cidx[j - 1], cidx[j]] += 1
            #     for i in range(j):
            #         self.counter[cidx[i], cidx[j]] += 1
            # next_obj = objective
            obj_done = False
            if objective < self.num_exploration_policies or self.exploring[i_batch]:
                next_obj = objective
                if self.exploration_only:
                    rare_achievements = [c for c in cidx[self.num_exploration_policies:] if self.known_objectives[c] not in self.original_known_objectives]
                else:
                    # rare_achievements = [c for c in cidx[1:] if self.completion_counter[c] < 20 or self.completion_counter[c] / (self.total_games + 1) < 0.1]
                    rare_achievements = [c for c in cidx[self.num_exploration_policies:] if self.completion_counter[c] < 20]
                obj_reward += 0.9 * len(rare_achievements) + 0.0 * len(step_completed)
            else:
                next_obj = self._get_next_objective(completed, self.objective_map[step_completed[-1]])
                if next_obj < self.num_exploration_policies and self.np_random.rand() < 0.0:
                    next_obj = objective
                    self.exploring[i_batch] = True
                if self.known_objectives[objective] in step_completed:
                    obj_reward += 100.
                    obj_done = True
                else:
                    obj_reward += 0.0
                    obj_done = True
                # obj_reward += 0.1
            if obj_done:
                self.intrinsic_reward[i_batch] = 0
            return next_obj, obj_reward, obj_done
        else:
            return objective, obj_reward, False
        
    def get_init_objective(self, i_batch):
        self.exploring[i_batch] = 0
        return self._get_next_objective([], 0)
    
    # def print(self):
    #     super().print()
        
    #     n = self.num_objectives
    #     on = self.original_counter.shape[0]

    #     print("")
    #     print("Direct counter:")
    #     dc = self.direct_counter.copy()
    #     dc[:on, :on] -= self.original_direct_counter
    #     for i in range(n):
    #         for j in range(n):
    #             print(dc[i, j], end=',' if j < n - 1 else '\n')

    #     print("")
    #     print("Counter:")
    #     c = self.counter.copy()
    #     c[:on, :on] -= self.original_counter
    #     for i in range(n):
    #         for j in range(n):
    #             print(c[i, j], end=',' if j < n - 1 else '\n')


class TargetObjectiveSelector(NewEmpiricalObjectiveSelector):
    def __init__(self, target, **kwargs):
        super().__init__(**kwargs)
        self.str_target = target

    def load(self, states):
        super().load(states)
        
        self.target = self.objective_map[self.str_target]

        for obj in self.known_objectives:
            print(obj)

        self._update_dep_graph(force_update=True)

        print(self.completion_counter)
        print(self.counter)
        print(self.direct_counter)

        self.required = set([self.target])
        for i in range(self.num_exploration_policies, self.num_objectives):
            if self.counter[i, self.target] >= self.completion_counter[self.target] * 0.8:
                print(self.known_objectives[i], self.counter[i, self.target] / self.completion_counter[self.target])
                self.required.add(i)

        print("Required objectives:")
        for i in self.required:
            print(self.known_objectives[i])

    def _get_next_objective(self, completed, last_completed):
        if self.target in completed:
            return 0
        choices = []
        weights = []
        id_completed = set([self.objective_map[c] for c in completed])
        id_last_completed = last_completed if last_completed == 0 else self.objective_map[last_completed]
        for i in range(self.num_exploration_policies, self.num_objectives):
            if i in self.required and self.known_objectives[i] not in id_completed and all([u in id_completed for u in range(self.num_exploration_policies, self.num_objectives) if self.dep_graph[u, i]]):
                choices.append(i)
                weights.append(self.direct_counter[id_last_completed, i] / self.completion_counter[i])
        weights = np.array(weights) + 1e-3
        if len(choices) == 0:
            return 0
        # next_obj = self.np_random.choice(choices, p=weights / weights.sum())
        next_obj = choices[np.argmax(weights)]
        print("next obj:", self.known_objectives[next_obj])
        return next_obj
        
    def get_next_objective(self, i_batch, obs, reward, done, info, completed, objective, step_completed, step_completed_full):
        if len(step_completed) > 0:
            return self._get_next_objective(completed, step_completed[-1]), 0, False
        else:
            return objective, 0, False
    
    def get_init_objective(self, i_batch):
        return self._get_next_objective([], 0)


OBJECTIVE_SELECTORS = {
    'dummy': lambda **kwargs: DummyObjectiveSelector(**kwargs),
    'dummyseq': lambda **kwargs: DummySequenceObjectiveSelector(**kwargs),
    'dummydense': lambda **kwargs: DummyDenseObjectiveSelector(**kwargs),
    'random': lambda **kwargs: RandomObjectiveSelector(**kwargs),
    'empirical': lambda **kwargs: NewEmpiricalObjectiveSelector(**kwargs),
    'exploration': lambda **kwargs: NewEmpiricalObjectiveSelector(exploration_only=True, **kwargs),
    'target': lambda **kwargs: TargetObjectiveSelector(**kwargs),
}


class ObjectiveSelectorOld:
    def __init__(self, num_objectives):
        self.num_objectives = num_objectives
        self.np_random = np.random.RandomState()
        self.objectives = None

    def seed(self, seed):
        self.np_random.seed(seed)

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.completed_objectives = np.zeros((self.batch_size, self.num_objectives), dtype=int)
        self.set_init_objectives()
        return self.objectives
    
    def get_step_completed(self, step_completed):
        return (1 - self.completed_objectives) & step_completed

    def step(self, step_completed, is_done):
        # step_completed_objectives: [batch_size, num_objectives] 0/1
        self.completed_objectives = self.completed_objectives | step_completed
        self.set_next_objectives(step_completed, is_done)
        return self.objectives

    def set_init_objectives(self):
        self.objectives = self.np_random.randint(self.num_objectives, size=self.batch_size)
    
    def set_next_objectives(self, step_completed, is_done):
        completed_anything = np.sum(step_completed, 1) > 0
        for i in range(self.batch_size):
            if is_done[i]:
                self.objectives[i] = self.np_random.randint(self.num_objectives)
                self.completed_objectives[i, :] = 0
            elif completed_anything[i]:
                uncompleted_objectives = [j for j in range(self.num_objectives) if not self.completed_objectives[i, j]]
                if len(uncompleted_objectives) == 0:
                    uncompleted_objectives = [0]
                self.objectives[i] = self.np_random.choice(uncompleted_objectives)


class SequentialObjectiveSelectorOld(ObjectiveSelectorOld):
    def set_init_objectives(self):
        self.objectives = np.zeros((self.batch_size,), dtype=int)

    def set_next_objectives(self, step_completed, is_done):
        completed_anything = np.sum(step_completed, 1) > 0
        for i in range(self.batch_size):
            if is_done[i]:
                self.objectives[i] = 0
                self.completed_objectives[i, :] = 0
            elif completed_anything[i]:
                uncompleted_objectives = [j for j in range(self.num_objectives) if not self.completed_objectives[i, j]]
                if len(uncompleted_objectives) == 0:
                    uncompleted_objectives = [0]
                self.objectives[i] = uncompleted_objectives[0]


def to_achv_str(numpy_achv, achv_list):
    return [achv_list[idx] for idx in numpy_achv.nonzero()]


class SEAVecWrapper(IVecEnvWrapper):
    def __init__(self, env: IVecEnv, objective_dim, objective_selector, objective_selector_load_path=None, objective_one_hot=False, objective_selector_config={}):
        super().__init__(env)
        self.objective_dim = objective_dim
        self.is_dict_obs = isinstance(env.get_env_info()["observation_space"], gym.spaces.Dict)
        self.np_random = np.random.RandomState(1)
        if objective_one_hot:
            self.objective_embed_book = np.eye(MAX_OBJECTIVES, self.objective_dim)
        else:
            self.objective_embed_book = self.np_random.randn(MAX_OBJECTIVES, self.objective_dim)
        print(self.objective_embed_book[:5, :5])
        print(objective_selector_config)
        self.objective_selector = OBJECTIVE_SELECTORS[objective_selector](**objective_selector_config)
        if objective_selector_load_path is not None:
            objective_states = torch_ext.load_checkpoint(objective_selector_load_path)
            try:
                self.objective_embed_book = objective_states.pop("objective_embed_book")
            except KeyError:
                pass
            self.objective_selector.load(objective_states)

        self.target_distances = TARGET_DISTANCES
        self.is_list_info = True

        print(self.target_distances)

    def get_selector_states(self):
        states = self.objective_selector.export()
        return {
            "objective_embed_book": self.objective_embed_book,
            **states
        }

    def seed(self, seed):
        self.np_random.seed(seed)
        self.objective_selector.seed(seed)
        self.env.seed(seed)

    def get_env_info(self):
        env_info = self.env.get_env_info()
        obs_space = env_info.pop("observation_space")
        if self.is_dict_obs:
            obs_space = gym.spaces.Dict({
                "objective": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.objective_dim,), dtype=float),
                **obs_space
            })
        else:
            obs_space = gym.spaces.Dict({
                "objective": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.objective_dim,), dtype=float),
                "observation": obs_space
            })
        return {
            "observation_space": obs_space,
            **env_info
        }
    
    # def _calc_progress(self, x_pos):
    #     t = [0] + self.target_distances
    #     pos = bisect.bisect(t, x_pos)
    #     return (x_pos - t[pos]) / (t[pos + 1] - t[pos])

    def _calc_pos(self, obs):
        x_pos = obs[0]
        return bisect.bisect(self.target_distances, x_pos)
    
    def _post_obs(self, obs, achv):
        # for i in range(self.batch_size):
        #     obs[i][0] = self._calc_pos(obs[i])
            # obs[i][0] = achv[i]
        return obs
    
    def _add_objective(self, obs):
        batch_size = self.batch_size
        # self.objective = self.np_random.randint(self.num_objectives, size=batch_size)
        num_objectives = self.objective_selector.num_objectives
        objective_oh = np.zeros((batch_size, num_objectives), dtype=float)
        objective_oh[np.arange(batch_size), self.objective] = 1  # one_hot
        # print(self.objective)
        embed = np.matmul(objective_oh, self.objective_embed_book[:num_objectives])
        if self.is_dict_obs:
            return {
                "objective": embed,
                **obs
            }
        else:
            return {
                "objective": embed,
                "observation": obs
            }

    def _get_batch_size(self, obs):
        if self.is_dict_obs:
            self.batch_size = obs[list(obs.keys())[0]].shape[0]
        else:
            self.batch_size = obs.shape[0]
    
    def reset(self):
        obs = self.env.reset()
        self._get_batch_size(obs)
        self.objective = self.objective_selector.reset(self.batch_size, obs)
        # print("self.objective", self.objective)
        return self._add_objective(self._post_obs(obs, [0] * self.batch_size))
    
    def _convert_to_dict_info(self, info):
        if type(info) == dict:
            return info
        else:
            dict_info = dict()
            for i in range(self.batch_size):
                for k, v in info[i].items():
                    if k not in dict_info:
                        dict_info[k] = [None] * self.batch_size
                    dict_info[k][i] = v
            return dict_info
        
    def _convert_to_list_info(self, info):
        if type(info) == list:
            return info
        else:
            def retrieve(d, i):
                if type(d) == dict:
                    ret = dict()
                    for k, v in d.items():
                        ret[k] = retrieve(v, i)
                    return ret
                elif type(d) == list and len(d) == self.batch_size:
                    return d[i]
                elif isinstance(d, torch.Tensor) and len(d.shape) > 0 and d.shape[0] == self.batch_size:
                    return d[i]
                else:
                    return None
            list_info = []
            for i in range(self.batch_size):
                list_info.append(retrieve(info, i))
            return list_info
    
    def step(self, actions):
        next_obs, reward, is_done, info = self.env.step(actions)

        # print(len(info))

        # print(info)

        # self.is_list_info = type(info) == list
        dict_info = self._convert_to_dict_info(info)
        # list_info = self._convert_to_list_info(info)

        # print(info['achievement_dense_rewards'].shape)
        # print(list_info[0])

        # print(info[0])

        step_completed = [to_achv_str(dict_info["unlocked"][i], info["achievement_list"]) for i in range(self.batch_size)]
        # for i in range(self.batch_size):
        #     if len(step_completed[i]) > 0:
        #         print(step_completed[i])

        # print(step_completed)

        # step_completed = [[] for _ in range(self.batch_size)]

        # for i in range(self.batch_size):
        #     x_pos = info['x_position'][i]
        #     pos = bisect.bisect(self.target_distances, x_pos)
        #     # pos = min(pos, 5)
        #     step_completed[i] = [f'target-{p}' for p in range(pos)]

        # print(step_completed, info['x_position'])
        # print(info["reward_ctrl"])
        # print(info['x_position'])
        # print(next_obs[:, 0])

        step_completed, step_completed_full = self.objective_selector.get_step_completed(step_completed)
        # obj_reward = np.zeros_like(reward)
        # for i in range(self.batch_size):
        #     obj_reward[i] = len(step_completed[i])
        achvs = []
        for i in range(self.batch_size):
            achvs.append(len(step_completed[i]) > 0)

        # print(info)

        # print(list_info[0])
        
        self.objective, obj_reward, obj_done = self.objective_selector.step(step_completed, step_completed_full, next_obs, reward, is_done, info)

        # print(self.objective, obj_reward, obj_done, step_completed, info['x_position'])

        return self._add_objective(self._post_obs(next_obs, achvs)), \
            obj_reward, \
            obj_done, \
            {
                "env_reward": reward,
                "env_is_done": is_done,
                "step_completed": step_completed,
                **dict_info
            }


class SEAWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, objective_dim, objective_selector, objective_selector_load_path=None, objective_one_hot=False, objective_selector_config={}):
        super().__init__(env)
        self.objective_dim = objective_dim
        self.is_dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
        self.np_random = np.random.RandomState(1)
        if objective_one_hot:
            self.objective_embed_book = np.eye(MAX_OBJECTIVES, self.objective_dim)
        else:
            self.objective_embed_book = self.np_random.randn(MAX_OBJECTIVES, self.objective_dim)
        print(self.objective_embed_book[:5, :5])
        # self.objective_selector = SequentialObjectiveSelector()
        # self.objective_selector.fill_objectives(50)
        self.objective_selector = OBJECTIVE_SELECTORS[objective_selector](**objective_selector_config)
        if objective_selector_load_path is not None:
            objective_states = torch_ext.load_checkpoint(objective_selector_load_path)
            try:
                self.objective_embed_book = objective_states.pop("objective_embed_book")
            except KeyError:
                pass
            self.objective_selector.load(objective_states)

        self.target_distances = TARGET_DISTANCES

    def get_selector_states(self):
        states = self.objective_selector.export()
        return {
            "objective_embed_book": self.objective_embed_book,
            **states
        }

    def seed(self, seed):
        self.np_random.seed(seed)
        self.objective_selector.seed(seed)
        self.env.seed(seed)

    @property
    def observation_space(self):
        obs_space = self.env.observation_space
        if self.is_dict_obs:
            obs_space = gym.spaces.Dict({
                "objective": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.objective_dim,), dtype=float),
                **obs_space
            })
        else:
            obs_space = gym.spaces.Dict({
                "objective": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.objective_dim,), dtype=float),
                "observation": obs_space
            })
        return obs_space
    
    def _calc_pos(self, obs):
        x_pos = obs[0]
        return bisect.bisect(self.target_distances, x_pos)
    
    def _post_obs(self, obs, achv):
        # obs[0] = self._calc_pos(obs)
        # obs[0] = int(achv)
        return obs
    
    def _add_objective(self, obs):
        num_objectives = self.objective_selector._num_objectives
        # print(num_objectives)
        objective_oh = np.zeros((1, num_objectives), dtype=float)
        objective_oh[0, self.objective] = 1  # one_hot
        embed = np.matmul(objective_oh, self.objective_embed_book[:num_objectives])[0]
        if self.is_dict_obs:
            return {
                "objective": embed,
                **obs
            }
        else:
            return {
                "objective": embed,
                "observation": obs
            }
    
    def reset(self):
        obs = self.env.reset()
        self.objective = self.objective_selector.reset(1)[0]
        # print(obs[0])
        return self._add_objective(self._post_obs(obs, 0))
    
    def step(self, actions):
        next_obs, reward, is_done, info = self.env.step(actions)

        # x_pos = info['x_position']
        # pos = bisect.bisect(self.target_distances, x_pos)
        # pos = min(pos, 5)
        # step_completed = [f'target-{i}' for i in range(pos)]
        # step_completed = list(range(pos))
        # print(next_obs[0], x_pos)

        # step_completed = list(info['unlocked'])
        step_completed = to_achv_str(info["unlocked"][0], info["achievement_list"])

        # print(step_completed)
        # print(info["unlocked"])

        step_completed, step_completed_full = self.objective_selector.get_step_completed([step_completed])
        # obj_reward = len(step_completed[0])

        self.objective, obj_reward, _ = self.objective_selector.step(step_completed, step_completed_full, np.array([is_done]))
        # print(self.objective, x_pos, step_completed, obj_reward, actions)

        # print(self.objective, obj_reward, step_completed)

        # return self._add_objective(self._post_obs(next_obs, len(step_completed) > 0)), obj_reward[0] + 0.000 * info["reward_ctrl"], is_done, info
        return self._add_objective(self._post_obs(next_obs, len(step_completed) > 0)), obj_reward[0], is_done, {
            "env_reward": reward,
            "env_is_done": is_done,
            "step_completed": step_completed[0],
            **info
        }


class SEAVecWrapperOld(IVecEnvWrapper):
    def __init__(self, env: IVecEnv, num_objectives):
        super().__init__(env)
        self.num_objectives = num_objectives
        self.is_dict_obs = isinstance(env.get_env_info()["observation_space"], gym.spaces.Dict)
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)
        self.env.seed(seed)

    def get_env_info(self):
        env_info = self.env.get_env_info()
        obs_space = env_info.pop("observation_space")
        if self.is_dict_obs:
            obs_space = gym.spaces.Dict({
                "objective": gym.spaces.Box(low=0, high=1, shape=(self.num_objectives,), dtype=int),
                **obs_space
            })
        else:
            obs_space = gym.spaces.Dict({
                "objective": gym.spaces.Box(low=0, high=1, shape=(self.num_objectives,), dtype=int),
                "observation": obs_space
            })
        return {
            "observation_space": obs_space,
            **env_info
        }
    
    def _add_objective(self, obs):
        batch_size = self.batch_size
        # self.objective = self.np_random.randint(self.num_objectives, size=batch_size)
        objective_np = np.zeros((batch_size, self.num_objectives), dtype=int)
        objective_np[np.arange(batch_size), self.objective] = 1  # one_hot
        if self.is_dict_obs:
            return {
                "objective": objective_np,
                **obs
            }
        else:
            return {
                "objective": objective_np,
                "observation": obs
            }

    def _get_batch_size(self, obs):
        if self.is_dict_obs:
            self.batch_size = obs[list(obs.keys())[0]].shape[0]
        else:
            self.batch_size = obs.shape[0]
    
    def reset(self):
        obs = self.env.reset()
        self._get_batch_size(obs)
        self.objective = self.np_random.randint(2, size=self.batch_size)
        return self._add_objective(obs)
    
    def step(self, actions):
        next_obs, reward, is_done, info = self.env.step(actions)

        print(info)

        for i in range(self.batch_size):
            if self.objective[i]:
                reward[i] = -reward[i]
            if is_done[i]:
                self.objective[i] = self.np_random.randint(2)

        return self._add_objective(next_obs), reward, is_done, info


class SEAWrapperOld(gym.Wrapper):
    def __init__(self, env: gym.Env, num_objectives):
        super().__init__(env)
        self.num_objectives = num_objectives
        self.is_dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random.seed(seed)
        self.env.seed(seed)

    @property
    def observation_space(self):
        obs_space = self.env.observation_space
        if self.is_dict_obs:
            obs_space = gym.spaces.Dict({
                "objective": gym.spaces.Box(low=0, high=1, shape=(self.num_objectives,), dtype=int),
                **obs_space
            })
        else:
            obs_space = gym.spaces.Dict({
                "objective": gym.spaces.Box(low=0, high=1, shape=(self.num_objectives,), dtype=int),
                "observation": obs_space
            })
        return obs_space
    
    def _add_objective(self, obs):
        objective_np = np.zeros((self.num_objectives,), dtype=int)
        objective_np[self.objective] = 1  # one_hot
        if self.is_dict_obs:
            return {
                "objective": objective_np,
                **obs
            }
        else:
            return {
                "objective": objective_np,
                "observation": obs
            }
    
    def reset(self):
        obs = self.env.reset()
        self.objective = self.np_random.randint(2)
        return self._add_objective(obs)
    
    def step(self, actions):
        next_obs, reward, is_done, info = self.env.step(actions)

        if self.objective:
            reward = -reward
        if is_done:
            self.objective = self.np_random.randint(2)

        return self._add_objective(next_obs), reward, is_done, info
