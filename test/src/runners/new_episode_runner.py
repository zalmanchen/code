from numpy.lib.function_base import select
import torch as th

from envs import REGISTRY as env_REGISTRY
from functools import partial
from learners.oppnent_learner import Oppnent_Learner
from components.episode_buffer import EpisodeBatch
from components.priority_buffer import PrioritizedReplayBuffer
from episode_buffer.Priority_buffer import PrioritizedReplayBuffer as EpisodeBuffer
from modules.agents.oppnent import EnemiesAgent
import numpy as np
import torch.nn.functional as F

import  json

import pdb

from learners.q_learner import QLearner

class EpisodeRunner:
    def __init__(self, args, logger, scheme):
        self.args = args
        self.logger = logger
        
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        #buffer
        buffer_size = 2e2   
        hidden_dim = 128
        imitate_buffer = PrioritizedReplayBuffer(num = self.env.n_enemies,
                                            obs_shape = self.env.get_obs_size(),
                                            hidden_shape = hidden_dim,
                                            action_shape = self.env.n_actions,
                                            size= buffer_size)

        agent_buffer = EpisodeBuffer(num = self.env.n_agents,
                                            obs_shape = self.env.get_obs_size(),
                                            hidden_shape = hidden_dim,
                                            action_shape = self.env.n_actions,
                                            state_shape = self.env.get_state_size(),
                                            size= buffer_size)


        self.agent =QLearner(args=args, env=self.env, logger=logger, agent_buffer= agent_buffer,
                                imitate_buffer =imitate_buffer ,device =self.args.device )
        self.collect_interval = 40
        self.t_env = 0
        
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        #log the first run
        self.log_train_stats_t = -1000000


    def reset(self):
        self.env.reset()
        self.t = 0

    def run(self, test_mode = False):
        self.reset()
        terminated = False
        #episode_return = np.repeat(0, self.env.n_agents)
        self.agent.init_hidden()

        pre_data = {"agent_obs":[], "hidden_state":[], "avail_actions":[],"state":[], "agent_actions":[]}
        cur_data = {"agent_obs":[], "hidden_state":[], "avail_actions":[],"state":[], "agent_actions":[]}

        reward =0
        episode_info = {}
        timestep = 0
        buffer_info = {}

        select_agent_actions = th.empty(1, self.env.n_agents)
        agent_actions = th.empty(1, self.env.n_agents)

        enemy_actions = np.repeat(4, self.env.n_agents)
        while not terminated:
            timestep_info = []
            buffer_actions =[]

            agent_actions= self.agent.forward(th.tensor(self.env.get_obs())).squeeze()
            avail_actions = th.tensor(self.env.get_avail_actions())
            agent_actions[avail_actions == 0] = - float("inf")

            agent_actions =th.argmax(agent_actions,dim=-1).squeeze()
            reward, terminated, env_info = self.env.step(agent_actions)
            enemy_actions = self.env.get_enemy_actions()
            self.agent.imitate_train(self.env.get_anti_obs(),enemy_actions, th.tensor(self.env.get_avail_actions()))
            timestep_info.extend(self.agent.agent_actions.numpy().tolist())
            timestep_info.extend(enemy_actions())
            episode_info[timestep] = timestep_info
            if terminated:
                print(self.env.get_stats())
            timestep += 1
            self.t += 1
        
            # print("_______________________________________________________")
            # print([obs[-1] for obs in self.env.get_obs()])
            # print([obs[-1] for obs in self.env.get_anti_obs()])

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        
        if not test_mode:
            self.t_env += self.t
        
        #pdb.set_trace()
        op = json.dumps(episode_info)
        fo = open('./result.json', 'a+')
        fo.write(op) 
        fo.write('\n')
        fo.close()

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()