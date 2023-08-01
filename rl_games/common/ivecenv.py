class IVecEnv:
    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def has_action_masks(self):
        return False

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        pass

    def seed(self, seed):
        pass

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        pass

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    def set_env_state(self, env_state):
        pass

    @property
    def unwrapped(self):
        return self
    
    @property
    def observation_space(self):
        return self.get_env_info()["observation_space"]
    
    @property
    def action_space(self):
        return self.get_env_info()["action_space"]


class IVecEnvWrapper(IVecEnv):
    def __init__(self, env: IVecEnv):
        super().__init__()
        self.env = env
    
    def step(self, actions):
        return self.env.step(actions)
    
    def reset(self):
        return self.env.reset()

    def has_action_masks(self):
        return self.env.has_action_masks()
    
    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        return self.env.get_env_info()

    def seed(self, seed):
        return self.env.seed(seed)

    def set_train_info(self, env_frames, *args, **kwargs):
        return self.env.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self):
        return self.env.get_env_state()

    def set_env_state(self, env_state):
        return self.env.set_env_state(env_state)

    @property
    def unwrapped(self):
        return self.env.unwrapped