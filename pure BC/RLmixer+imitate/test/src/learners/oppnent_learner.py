from modules.agents.oppnent import EnemyAgent, EnemiesAgent
import matplotlib as plt
from sklearn import manifold
import torch as th
import numpy as np
import torch.nn.functional as F
from torch.optim import RMSprop

import json

import pdb

class Oppnent_Learner:
    def __init__(self, args, env, device=None, buffer=None, logger=None, interval=50, training_episode=10):

        self.env = env
        self.params = []
        env.reset()
        env_info = env.get_env_info()
        obs = env.get_anti_obs()
        self.args = args
        self.enemies = EnemiesAgent(env_info['n_agents'], obs, env_info['n_actions'])
        self.hidden_state = self.enemies.init_hidden()

        self.logger = logger
        self.tsne = manifold.TSNE(n_components=2, init='random', random_state=314, n_iter=300, learning_rate=100)
        self.imitate_list = list()
        self.params += list(self.enemies.parameters())
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.enemy_actions_softmax = list()

        # match-up parameters
        self.skip_counter = 0
        self.skip_frame = 15
        self.agents_match_up = None

        # oppnent training parameters
        self.interval = interval
        self.batch_size = 16
        self.episode = 0
        self.training_episode = training_episode
        self.device = device

        # to device
        # self.model2device()

        # replay buffer
        self.buffer = buffer

        # PER parameters
        self.prior_eps = 1e-4

        # enemies transition
        self.opponent_data = {"enemy_obs": [], "enemy_actions": [], "hidden_state": [], "avail_actions": []}
    
    def init_hidden(self):
        self.hidden_state = self.enemies.init_hidden()

    def train(self, enemy_obs, enemy_actions, avail_actions, n_actions):
        # Get the relevant quantities
        enemy_obs = th.FloatTensor(np.array(enemy_obs))
        enemy_actions = th.LongTensor(np.array([enemy_actions])).squeeze()
        #enemy_actions = F.one_hot(enemy_actions, n_actions)

        self.opponent_data["enemy_obs"] = enemy_obs.unsqueeze(1)
        self.opponent_data["enemy_actions"] = enemy_actions.unsqueeze(1)
        self.opponent_data["hidden_state"] = th.cat(self.hidden_state, dim=0).unsqueeze(1)
        self.opponent_data["avail_actions"] = avail_actions.squeeze()

        enemy_actions_predict, self.hidden_state = self.enemies.forward(enemy_obs, self.hidden_state)
        enemy_actions_predict = enemy_actions_predict.squeeze()

        enemy_actions_predict[avail_actions == 0] = -999999      
        self.enemy_actions_softmax = F.softmax (enemy_actions_predict, dim=-1).squeeze()

        CE  =  th.nn.CrossEntropyLoss()
        CE_loss =CE(self.enemy_actions_softmax, enemy_actions)
        kl_prior =CE_loss.cpu().item()

        #kl_prior shoule be the CrossEntropyLoss 
        #kl_prior = F.kl_div(self.enemy_actions_softmax, enemy_actions.float(), reduction="sum").detach().abs().cpu().item()

        self.buffer.store(self.opponent_data, loss_prior=[kl_prior])
        self.episode += 1
        # training part
        # store the enemy_actions by json
        loss_info = {}

        if self.episode % self.interval == 0:

            for step in range(self.training_episode):
                data = self.buffer.sample_batch()

                avail_actions = th.FloatTensor(data["avail_actions"]).squeeze()
                enemy_obs = th.FloatTensor(data["enemy_obs"]).squeeze().transpose(0, 1)  # .to(self.device)
                enemy_actions = th.FloatTensor(data["enemy_actions"]).squeeze() # .to(self.device)
            

                # print("-----------------------------------------------------------------------")
                # print(enemy_actions.shape)
                hidden_state = th.FloatTensor(data["hidden_state"]).squeeze().transpose(0, 1)  # .to(self.device)
                weights = th.FloatTensor(data["weights"].astype(np.float32).reshape(-1, 1))# .to(self.device)
                indices = data["indices"]
                enemy_actions_predict, _ = self.enemies.forward(enemy_obs, hidden_state)
                enemy_actions_predict = enemy_actions_predict.squeeze()

                enemy_actions_predict[avail_actions == 0] = -float("inf")
                enemy_actions_predict = F.softmax(enemy_actions_predict, dim=-1)

                loss_info[self.episode+step]=th.argmax(enemy_actions_predict, dim=-1).tolist() # store the enemy_actions  into the loss_info
                # enemy_actions_index = th.argmax(enemy_actions_predict, dim=-1)
                # enemy_actions_one_hot = F.one_hot(enemy_actions_index, n_actions)
                # enemy_actions_predict.mul(enemy_actions_one_hot)
                # error = enemy_actions_predict - enemy_actions

                # calculate CE_Loss

                CE  =  th.nn.CrossEntropyLoss(reduction = 'none',ignore_index = 0)
                CE_loss =CE(enemy_actions_predict.transpose(1,2), enemy_actions.type(th.LongTensor))
                CE_sum=(CE_loss.sum(dim=1)*weights.squeeze()).mean()

                self.optimiser.zero_grad()
                CE_sum.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip) #max_norm
                self.optimiser.step()

                loss_for_piror = CE_loss.detach().sum(dim=1).abs().cpu().numpy()
                new_priorities = loss_for_piror + self.prior_eps
                new_priorities = new_priorities.tolist()
                self.buffer.update_priorities(indices, new_priorities)

            op = json.dumps(loss_info)
            fo = open('./sample.json', 'a+')
            fo.write(op) 
            fo.write('\n')
            fo.close()

                # calculate KL divergence
                
                # kl_divergence = F.kl_div(enemy_actions_predict, enemy_actions, reduction="none").sum(dim=-1)
                # kl_sum = (kl_divergence.sum(dim=0) * weights.squeeze()).mean()

                # Optimise
                # self.optimiser.zero_grad()
                # kl_sum.backward()
                # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
                # self.optimiser.step()

                # PER update priorities
                # loss_for_piror = kl_divergence.detach().sum(dim=0).abs().cpu().numpy()
                # new_priorities = loss_for_piror + self.prior_eps
                # new_priorities = new_priorities.tolist()
                # self.buffer.update_priorities(indices, new_priorities)


    def model2device(self):
        self.enemies.to(self.device)

    def save_models(self, path):
        self.enemies.save_models(path)

    def load_models(self, path):
        self.enemies.load_models(path)


    def match_up(self, agents, enemies):
        self.skip_counter += 1

        if self.skip_counter % self.skip_frame == 0:
            n_enemies = len(enemies)
            n_agents = len(agents)
            agent_obs = list()
            enemy_obs = list()
            agents = agents.copy()
            enemies = enemies.copy()

            # enemies position
            for e_id in range(n_enemies):
                center_x = enemies[e_id].pos.x
                center_y = enemies[e_id].pos.y
                enemy_obs_id = list()
                for e in range(n_enemies):
                    if enemies[e].health == 0.:
                        enemy_obs_id += [0., 0., 0.]
                    else:
                        enemy_obs_id.append(enemies[e].health)
                        enemy_obs_id.append(enemies[e].pos.x - center_x)
                        enemy_obs_id.append(enemies[e].pos.y - center_y)
                for a in range(n_agents):
                    if agents[a].health == 0.:
                        enemy_obs_id += [0., 0., 0.]
                    else:
                        enemy_obs_id.append(agents[a].health)
                        enemy_obs_id.append(agents[a].pos.x - center_x)
                        enemy_obs_id.append(agents[a].pos.y - center_y)
                enemy_obs.append(enemy_obs_id)

            # agent position
            for a_id in range(n_agents):
                center_x = agents[a_id].pos.x
                center_y = agents[a_id].pos.y
                agent_obs_id = list()
                for a in range(n_agents):
                    if agents[a].health == 0.:
                        agent_obs_id += [0., 0., 0.]
                    else:
                        agent_obs_id.append(agents[a].health)
                        agent_obs_id.append(agents[a].pos.x - center_x)
                        agent_obs_id.append(agents[a].pos.y - center_y)
                for e in range(n_enemies):
                    if enemies[e].health == 0.:
                        agent_obs_id += [0., 0., 0.]
                    else:
                        agent_obs_id.append(enemies[e].health)
                        agent_obs_id.append(enemies[e].pos.x - center_x)
                        agent_obs_id.append(enemies[e].pos.y - center_y)
                agent_obs.append(agent_obs_id)

            labels = np.array([0 for i in range(len(agent_obs))] + [1 for i in range(len(enemy_obs))])
            agents = np.array(agent_obs + enemy_obs)
            agents_match_up = self.tsne.fit_transform(agents)

            # if render:
                # x_min, x_max = agents_match_up.min(0), agents_match_up.max(0)
                # match_up_norm = (agents_match_up - x_min) / (x_max - x_min)
                # plt.figure(figsize=(8, 8))
                # for i in range(match_up_norm.shape[0]):
                #     plt.text(match_up_norm_norm[i, 0], match_up_norm_norm[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i]),
                #              fontdict={'weight': 'bold', 'size': 9})
                # plt.xticks([])
                # plt.yticks([])
                # plt.show()

            self.imitate_list = list()
            for a_id in range(len(agent_obs)):
                min_dis = 9999999.
                for e_id in range(len(enemy_obs)):
                    index = len(enemy_obs) - 1
                    dist = np.linalg.norm(agents_match_up[a_id] - agents_match_up[len(agent_obs) + e_id])
                    if min_dis > dist:
                        min_dis = dist
                        index = e_id
                self.imitate_list.append(index)

    def imitation(self, agents, enemies, agent_actions, timestep_info):
        self.match_up(agents, enemies)
        imitate_reward = list()
        agent_actions = agent_actions.squeeze()
        timestep_info.extend(th.argmax(self.enemy_actions_softmax, dim=-1).squeeze().tolist())

        for id in range(len(self.imitate_list)):
            enemy_act = self.enemy_actions_softmax.squeeze()[self.imitate_list[id]] + th.tensor(1e-5)
            agent_act = agent_actions[id] + th.tensor(1e-5)


            # info reward
            #info = th.distributions.Categorical(enemy_act)
           # entropy_reward = -info.entropy().abs()
           # print("agent actions", agent_act)
           # print("enemy actions",enemy_act)

            imitate_reward.append(th.exp(F.kl_div(agent_act, enemy_act )).clone().detach()-1 )
                                   #\ + th.exp(entropy_reward).clone().detach() - 1).cpu().item())
            #imitate_reward.append(th.exp(kl_div(agent_actions[id], enemy_act)).detach())
            #print(F.kl_div(agent_act, enemy_act))
        #print([ir.item() for ir in imitate_reward])

        # if imitate list is empty
        if not self.imitate_list:
            imitate_reward = [0 for i in range(len(enemies))]

        return imitate_reward