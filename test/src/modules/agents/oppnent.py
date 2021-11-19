import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pdb

class EnemyAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnemyAgent, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        x = F.relu(self.fc1(inputs))
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        h = self.rnn(x, hidden_state)
        q = self.fc2(x)
        return q, h

class EnemiesAgent(nn.Module):
    def __init__(self, n_agents, obs, n_actions):
        super(EnemiesAgent, self).__init__()
        
        self.obs = obs
        self.n_agents = n_agents
        self.output_dim = n_actions
        self.rnn_agents = nn.ModuleList()
        
        for i in range(self.n_agents):
            self.rnn_agents.append(EnemyAgent(len(obs[i]), 128, n_actions))

    def init_hidden(self):
        init_ret = []
        for agent in self.rnn_agents:
            init_ret.append(agent.init_hidden())
        return init_ret

    def forward(self, obs, hidden_states):
        q_list = []
        h_list = []
        for i in range(self.n_agents):
            q, h = self.rnn_agents[i](obs[i], hidden_states[i])
            q_list.append(q)
            h_list.append(h)

        #return th.stack(q_list, dim=1), th.stack(h_list, dim=1)
        return th.stack(q_list, dim=1), h_list

