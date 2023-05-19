import numpy as np
import gym
import bisect

from rl_games.common.ivecenv import IVecEnv, IVecEnvWrapper


MAX_OBJECTIVES = 50
START = 0.1
STEP = 1.5
# TARGET_DISTANCES = [0.1 * i for i in range(20)] + [2 * 1.2 ** i for i in range(30)]
TARGET_DISTANCES = [1 * 1.5 ** i for i in range(50)]
# TARGET_DISTANCES = [i + 1 for i in range(50)]


class ObjectiveSelector:
    def __init__(self):
        self.num_objectives = 1  # obj-0 is exploration
        self.known_objectives = ['explore']
        self.objective_map = {'explore': 0}
        self.np_random = np.random.RandomState()
        self.objectives = None

    def seed(self, seed):
        self.np_random.seed(seed)

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.completed_objectives = [[] for _ in range(batch_size)]
        self.objectives = np.zeros((batch_size,), dtype=int)
        for i in range(batch_size):
            self.objectives[i] = self.get_init_objective()
        return self.objectives
    
    def get_step_completed(self, step_completed):
        ret_step_completed = [[] for _ in range(self.batch_size)]
        updated = False
        for i in range(self.batch_size):
            for c in step_completed[i]:
                if c not in self.completed_objectives[i]:
                    if c not in self.known_objectives:
                        self.known_objectives.append(c)
                        self.objective_map[c] = self.num_objectives
                        self.num_objectives += 1
                        updated = True
                    self.completed_objectives[i].append(c)
                    ret_step_completed[i].append(c)
        if updated:
            self.update_known_objectives()
        return ret_step_completed

    def step(self, step_completed, is_done):
        obj_done = np.copy(np.array(is_done))
        obj_rew = np.zeros_like(is_done, dtype=float)
        for i in range(self.batch_size):
            if is_done[i]:
                self.objectives[i] = self.get_init_objective()
                self.completed_objectives[i] = []
            else:
                self.objectives[i], obj_rew[i], obj_done[i] = self.get_next_objective(self.completed_objectives[i], self.objectives[i], step_completed[i])
        return self.objectives, obj_rew, obj_done
    
    def update_known_objectives(self):
        pass

    def get_init_objective(self):
        pass
    
    def get_next_objective(self, completed, objective, step_completed):
        pass
    

class RandomObjectiveSelector(ObjectiveSelector):
    def set_num_objectives(self, num_objectives):
        self._num_objectives = num_objectives
    
    def get_init_objective(self):
        return self.np_random.randint(self._num_objectives)
    
    def get_next_objective(self, completed, objective, step_completed):
        return self.np_random.randint(self._num_objectives), len(step_completed), False
    

class SequentialObjectiveSelector(ObjectiveSelector):
    def fill_objectives(self, num_objectives):
        for i in range(self.num_objectives, num_objectives + 1):
            self.known_objectives.append(i - 1)
            self.objective_map[i - 1] = i
        self.num_objectives = num_objectives

    def get_init_objective(self):
        return 1 if self.num_objectives > 1 else 0
    
    def get_next_objective(self, completed, objective, step_completed):
        if len(step_completed) > 0:
            return min(max([self.objective_map[c] for c in step_completed]) + 1, self.num_objectives - 1), len(step_completed), False
        else:
            return objective, 0, False
        

class EmpiricalObjectiveSelector(ObjectiveSelector):
    def __init__(self):
        super().__init__()
        self.counter = None
        self.direct_counter = None

    def reset(self, batch_size):
        super().reset(batch_size)
        self.counter = None
        self.direct_counter = None

    def update_known_objectives(self):
        n = self.num_objectives
        new_counter = np.zeros((n, n), dtype=int)
        new_direct_counter = np.zeros((n, n), dtype=int)
        if self.counter is not None:
            _n = self.counter.shape[0]
            new_counter[:_n, :_n] = self.counter
            new_direct_counter[:_n, :_n] = self.direct_counter
        self.counter = new_counter
        self.direct_counter = new_direct_counter

    def _get_next_objective(self, last_completed):
        if self.direct_counter is None:
            return 0
        else:
            selection = np.argmax(self.direct_counter[last_completed])
            if self.direct_counter[last_completed, selection] == 0:
                selection = 0
            # print('next', last_completed, selection)
            return selection

    def get_next_objective(self, completed, objective, step_completed):
        if len(step_completed) > 0:
            cidx = [0] + [self.objective_map[completed[i]] for i in range(len(completed))]
            for j in range(len(completed) - len(step_completed) + 1, len(cidx)):
                self.direct_counter[cidx[j - 1], cidx[j]] += 1
                for i in range(j):
                    self.counter[cidx[i], cidx[j]] += 1
            next_obj = self._get_next_objective(self.objective_map[step_completed[-1]])
            obj_done = False
            if objective == 0:
                obj_reward = len(step_completed)
            else:
                if self.known_objectives[objective] in step_completed:
                    obj_reward = 1.
                    obj_done = True
                else:
                    obj_reward = 0.
            return next_obj, obj_reward, obj_done
        else:
            return objective, 0., False
        
    def get_init_objective(self):
        return self._get_next_objective(0)


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


class SEAVecWrapper(IVecEnvWrapper):
    def __init__(self, env: IVecEnv, objective_dim):
        super().__init__(env)
        self.objective_dim = objective_dim
        self.is_dict_obs = isinstance(env.get_env_info()["observation_space"], gym.spaces.Dict)
        self.np_random = np.random.RandomState(1)
        self.objective_embed_book = np.random.randn(MAX_OBJECTIVES, self.objective_dim)
        print(self.objective_embed_book[:5, :5])
        self.objective_selector = EmpiricalObjectiveSelector()

        self.target_distances = TARGET_DISTANCES
        self.is_list_info = True

        print(self.target_distances)

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
        self.objective = self.objective_selector.reset(self.batch_size)
        return self._add_objective(self._post_obs(obs, [0] * self.batch_size))
    
    def _convert_to_list_info(self, info):
        if type(info) == list:
            return info
        else:
            infos = [dict() for _ in range(self.batch_size)]
            for k, v in info.items():
                if type(v) == list and len(v) == self.batch_size:
                    for i in range(self.batch_size):
                        infos[i][k] = v[i]
            return infos
    
    def step(self, actions):
        next_obs, reward, is_done, info = self.env.step(actions)
        # self.is_list_info = type(info) == list
        info = self._convert_to_list_info(info)

        step_completed = [list(info[i]["unlocked"]) for i in range(self.batch_size)]
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

        step_completed = self.objective_selector.get_step_completed(step_completed)
        # obj_reward = np.zeros_like(reward)
        # for i in range(self.batch_size):
        #     obj_reward[i] = len(step_completed[i])
        achvs = []
        for i in range(self.batch_size):
            achvs.append(len(step_completed[i]) > 0)
        
        self.objective, obj_reward, obj_done = self.objective_selector.step(step_completed, is_done)

        # print(self.objective, obj_reward, obj_done, step_completed, info['x_position'])

        return self._add_objective(self._post_obs(next_obs, achvs)), \
            obj_reward, \
            obj_done, \
            [{
                "env_reward": reward[i],
                "env_is_done": is_done[i],
                "step_completed": step_completed[i],
                **info[i]
            } for i in range(self.batch_size)]


class SEAWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, objective_dim):
        super().__init__(env)
        self.objective_dim = objective_dim
        self.is_dict_obs = isinstance(env.observation_space, gym.spaces.Dict)
        self.np_random = np.random.RandomState(1)
        self.objective_embed_book = np.random.randn(MAX_OBJECTIVES, self.objective_dim)
        print(self.objective_embed_book[:5, :5])
        # self.objective_selector = SequentialObjectiveSelector()
        # self.objective_selector.fill_objectives(50)
        self.objective_selector = RandomObjectiveSelector()
        self.objective_selector.set_num_objectives(8)

        self.target_distances = TARGET_DISTANCES

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

        step_completed = list(info['unlocked'])

        step_completed = self.objective_selector.get_step_completed([step_completed])
        # obj_reward = len(step_completed[0])

        self.objective, obj_reward, _ = self.objective_selector.step(step_completed, np.array([is_done]))
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
