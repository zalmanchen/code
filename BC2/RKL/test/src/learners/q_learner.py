import copy
from modules.agents.rnn_agent import RNNAgent, RNNAgents

from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np
import torch.nn.functional as F

import pdb

class QLearner:
    def __init__(self, args, env,  logger, device = None, agent_buffer=None, imitate_buffer=None,interval = 50, training_episode = 10):
        self.args = args
        self.logger = logger

        #self.params =[]
        #self.params += list(self.agent.parameters() )
        self.params =[]
        self.env = env
        env.reset()
        env_info = env.get_env_info()
        obs = env.get_obs()
         
        self.agent = RNNAgents( env_info['n_agents'], obs, env_info['n_actions'])
        self.hidden_state = self.agent.init_hidden()
        self.agent_hidden_state = copy.deepcopy(self.hidden_state)
        self.target_agent= copy.deepcopy(self.agent)
        self.params +=  list(self.agent.parameters())
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.agent_actions = []
        self. agent_actions_softmax = []

        self.last_target_update_episode = 0

        self.agent_data = {"agent_obs": [], "enemy_actions": [], "hidden_state": [], "avail_actions": []}
        self.mixer = QMixer(args)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.log_stats_t = -self.args.learner_log_interval - 1

        # opponent training parameters
        self.interval = interval
        self.batch_size = 16
        self.episode = 0
        self.training_episode = training_episode
        self.device = device

        self.agent_buffer = agent_buffer
        self.imitate_buffer = imitate_buffer
        # prioritized experience replay
        self.prior_eps = 1e-4


    def init_hidden(self):
        self.hidden_state = self.agent.init_hidden()
        self.agent_hidden_state = self.agent.init_hidden()
        # agent_actions: obtained by  self.mac.select_action()

    #cur_data =  {cur_agent_obs, cur_agent_actions, cur_avail_actions}
    #pre_data = {pre_agent_obs, pre_agent_actions, pre_avail_actions}

    def select_actions(self, agent_actions, avail_actions, t_env, test_mode = False):
        return self.action_selector.select_action(agent_actions, avail_actions, t_env, test_mode = test_mode)

    
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def forward(self, agent_obs):
        agent_actions_predict, self.agent_hidden_state = self.agent.forward(agent_obs, self.agent_hidden_state)
        return agent_actions_predict


    # notice: use enemy_obs and enemy_actions to train 
    def imitate_train(self, agent_obs, enemy_actions, avail_actions):
                # Get the relevant quantities
        agent_obs = th.FloatTensor(np.array(agent_obs))
        enemy_actions = th.LongTensor(np.array([enemy_actions])).squeeze()
        #enemy_actions = F.one_hot(enemy_actions, n_actions)

        self.agent_data["agent_obs"] = agent_obs.unsqueeze(1)
        self.agent_data["enemy_actions"] = enemy_actions.unsqueeze(1)
        self.agent_data["hidden_state"] = th.cat(self.hidden_state, dim=0).unsqueeze(1)
        self.agent_data["avail_actions"] = avail_actions.squeeze()

        agent_actions_predict, self.hidden_state = self.agent.forward(agent_obs, self.hidden_state)
        agent_actions_predict = agent_actions_predict.squeeze()
        agent_actions_predict[avail_actions == 0] = -float("inf")
        self.agent_actions_softmax = F.softmax (agent_actions_predict, dim=-1).squeeze()
        self.agent_actions =th.argmax(self.agent_actions_softmax, dim=-1)


        CE  =  th.nn.CrossEntropyLoss()
        CE_loss =CE(self.agent_actions_softmax, enemy_actions)
        kl_prior =CE_loss.cpu().item()

        
        #kl_prior shoule be the CrossEntropyLoss 
        #kl_prior = F.kl_div(self.enemy_actions_softmax, enemy_actions.float(), reduction="sum").detach().abs().cpu().item()


        self.imitate_buffer.store(self.agent_data, loss_prior=[kl_prior])
        self.episode += 1
        # training part
        # store the enemy_actions by json 

        if self.episode % self.interval == 0:

            for step in range(self.training_episode):
                data = self.imitate_buffer.sample_batch()

                avail_actions = th.FloatTensor(data["avail_actions"]).squeeze()
                agent_obs = th.FloatTensor(data["agent_obs"]).squeeze().transpose(0, 1)  # .to(self.device)
                enemy_actions = th.FloatTensor(data["enemy_actions"]).squeeze() # .to(self.device)

                # print("-----------------------------------------------------------------------")
                # print(enemy_actions.shape)
                hidden_state = th.FloatTensor(data["hidden_state"]).squeeze().transpose(0, 1)  # .to(self.device)
                weights = th.FloatTensor(data["weights"].astype(np.float32).reshape(-1, 1))# .to(self.device)
                indices = data["indices"]
                agent_actions_predict, _ = self.agent.forward(agent_obs, hidden_state)
                agent_actions_predict = agent_actions_predict.squeeze()

                
                agent_actions_predict[avail_actions == 0] = -float("inf")
                agent_actions_predict = F.softmax(agent_actions_predict, dim=-1)
                # store the enemy_actions  into the loss_info

                # enemy_actions_index = th.argmax(enemy_actions_predict, dim=-1)
                # enemy_actions_one_hot = F.one_hot(enemy_actions_index, n_actions)
                # enemy_actions_predict.mul(enemy_actions_one_hot)
                # error = enemy_actions_predict - enemy_actions

                # calculate CE_Loss
                CE  =  th.nn.CrossEntropyLoss(reduction = 'none')
                CE_loss =CE(agent_actions_predict.transpose(1,2), enemy_actions.type(th.LongTensor))
                CE_sum=(CE_loss.sum(dim=1)*weights.squeeze()).mean()

                self.optimiser.zero_grad()
                CE_sum.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip) #max_norm
                self.optimiser.step()

                loss_for_piror = CE_loss.detach().sum(dim=1).abs().cpu().numpy()
                new_priorities = loss_for_piror + self.prior_eps
                new_priorities = new_priorities.tolist()
                self.imitate_buffer.update_priorities(indices, new_priorities)