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
        buffer = PrioritizedReplayBuffer(num = self.env.n_enemies,
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


        self.agent =QLearner(args=args, env=self.env, logger=logger, buffer= agent_buffer, device =self.args.device )
        
        self.opponent = Oppnent_Learner(args = args, env = self.env,
                                                        buffer = buffer, device = self.args.device)
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
        episode_return = np.repeat(0, self.env.n_agents)
        self.agent.init_hidden()
        self.opponent.init_hidden()

        pre_data = {"agent_obs":[], "hidden_state":[], "avail_actions":[],"state":[], "agent_actions":[]}
        cur_data = {"agent_obs":[], "hidden_state":[], "avail_actions":[],"state":[], "agent_actions":[]}

        reward =0
        episode_info = {}
        timestep = 0
        buffer_info = {}

        select_agent_actions = th.empty(1, self.env.n_agents)
        agent_actions = th.empty(1, self.env.n_agents)

        enemy_actions =[]
        while not terminated:
            timestep_info = []
            buffer_actions =[]

            if self.env.get_stats()["battle_won"]  > 0: 
                pre_data["agent_obs"] = cur_data["agent_obs"]
                pre_data["agent_actions"] = cur_data["agent_actions"] # the output of the RNNAgents 
                pre_data["hidden_state"] = cur_data["hidden_state"]        # the output of the RNNAgents, and the inputs of the next step 
                pre_data["avail_actions"]  = cur_data["avail_actions"]
                pre_data["state"]  = cur_data["state"]
                agent_actions = select_agent_actions

                cur_data["avail_actions"] = th.tensor(self.env.get_avail_actions()).unsqueeze(0)
                cur_data["state"] = self.env.get_state()
                cur_data["agent_obs"] = self.env.get_obs()
                cur_data["hidden_state"] = self.agent.hidden_state 

                cur_agent_actions, self.agent.hidden_state=self.agent.forward(th.tensor(cur_data["agent_obs"]), self.agent.hidden_state)
                cur_data["agent_actions"] = cur_agent_actions
                cur_agent_actions[cur_data["avail_actions"] == 0] = -float("inf") # masked
                agent_actions_softmax = th.nn.functional.softmax(cur_agent_actions, dim=-1)
                #max_agent_actions =(th.argmax( cur_agent_actions, dim=-1)).squeeze().tolist()

                select_agent_actions= self.agent.select_actions(cur_agent_actions, 
                                            th.tensor(cur_data["avail_actions"]), self.t_env, test_mode)

                reward, terminated, env_info = self.env.step(select_agent_actions) #Q(S_cur, A_cur)

                timestep_info.extend(enemy_actions)
                self.opponent.train(th.tensor(enemy_obs), enemy_actions, enemy_avail_actions, self.env.n_actions)

                if terminated:# and self.t_env > 1000:
                    self.agent.train(cur_data, cur_data, select_agent_actions,reward, buffer_actions, terminated, t_env=self.t_env)
                    if  test_mode:
                        print(self.env.get_stats())
                elif timestep>0:# and self.t_env>1000:
                    self.agent.train(pre_data, cur_data, agent_actions,reward, 
                                                    buffer_actions =buffer_actions, terminated= terminated, t_env= self.t_env )

            if timestep ==0:
                self.agent.agent_actions = np.repeat(4, self.n_agents)
            else:
                self.agent.imitate_train(self.env.get_obs(), enemy_actions, self.env.get_avail_actions(),self.env.get_avail_actions())

            timestep_info.extend(self.agent.agent_actions.tolist())
            enemy_obs = self.env.get_anti_obs()
            pdb.set_trace()
            enemy_actions = self.env.get_enemy_actions()
            enemy_avail_actions = th.tensor(self.env.get_enemy_avail_actions())

            timestep_info.extend(enemy_actions)
            timestep_info.extend(self.agent.agent_actions)

            episode_info[timestep] = timestep_info
            timestep += 1
            self.t += 1

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