import random
import torch as th
from typing import Dict, List
import numpy as np
from .SegmentTree import MinSegmentTree, SumSegmentTree

import json
import pdb

# Normal ReplayBuffer for Multi-agent

class ReplayBuffer:
    """normal agent replay buffer."""
    def __init__(self, num: int,
                 obs_shape: int,
                 hidden_shape: int,
                 action_shape: int,
                 state_shape: int,
                 size: int,
                 batch_size: int = 16):

        self.max_size, self.batch_size = int(size), int(batch_size)
        state_shape=int(state_shape)
        self.ptr, self.size, = 0, 0

        self.state_buf = np.zeros([self.max_size, 1, state_shape], dtype=np.float32)
        self.obs_buf = np.zeros([self.max_size, num, obs_shape], dtype=np.float32)
        self.hidden_buf = np.zeros([self.max_size, num, hidden_shape], dtype=np.float32)
        self.avail_buf = np.zeros([self.max_size, num, action_shape], dtype=np.float32)

        self.pre_state_buf = np.zeros([self.max_size, 1, state_shape], dtype=np.float32)
        self.pre_obs_buf = np.zeros([self.max_size, num, obs_shape], dtype=np.float32)
        self.pre_hidden_buf = np.zeros([self.max_size, num, hidden_shape], dtype=np.float32)
        self.pre_acts_buf = np.zeros([self.max_size, num, 1], dtype=np.float32)
        self.pre_avail_buf = np.zeros([self.max_size, num, action_shape], dtype=np.float32)

        self.reward_buf = np.zeros([self.max_size, num, 1], dtype=np.float32)
        self.imitate_buf = np.zeros([self.max_size, num, 1], dtype=np.float32)
    def store(
            self,
            transitions
    ):

        self.pre_obs_buf[self.ptr] = transitions["pre_agent_obs"].detach().squeeze().cpu().numpy()
        self.pre_hidden_buf[self.ptr] = transitions["pre_hidden_state"].detach().squeeze().cpu().numpy()
        self.pre_acts_buf[self.ptr] = transitions["pre_agent_actions"].detach().squeeze().cpu().numpy()
        self.pre_avail_buf[self.ptr] = transitions["pre_avail_obs"].detach().squeeze().cpu().numpy()
        self.pre_state_buf[self.ptr] = transitions["pre_state"].detach().squeeze().cpu().numpy()
        
        self.reward_buf[self.ptr] = transitions["reward"].detach().squeeze().cpu().numpy()
        self.imitate_buf[self.ptr] = transitions["imitate_reward"].detach().squeeze().cpu().numpy()
    
        self.state_buf[self.ptr] = transitions["state"].detach().squeeze().cpu().numpy()
        self.avail_buf[self.ptr] = transitions["avail_actions"].detach().squeeze().cpu().numpy()
        self.obs_buf[self.ptr] = transitions["agent_obs"].detach().squeeze().cpu().numpy()
        self.hidden_buf[self.ptr] = transitions["hidden_state"].detach().squeeze().cpu().numpy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.size, replace=False)
        return {"agent_obs": self.obs_buf[idxs],
                "hidden_state": self.hidden_buf[idxs],
                "avail_actions": self.avail_buf[idxs],
                "state": self.state_buf[idxs],

                "reward": self.reward_buf[idxs],
                "imitate_reward": self.imitate_buf[idxs],

                "pre_state": self.pre_state_buf[idxs],
                "pre_avail_actions": self.pre_avail_buf[idxs],
                "pre_agent_obs": self.pre_obs_buf[idxs],
                "pre_hidden_state": self.pre_hidden_buf[idxs],
                "pre_agent_actions": self.pre_acts_buf[idxs],
                }

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """
    def __init__(
            self, num:int,
            obs_shape: int,
            hidden_shape: int,
            action_shape: int,
            state_shape:int,
            size: int,
            batch_size: int = 16,
            alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0

        self.num = num

        super(PrioritizedReplayBuffer, self).__init__(num, obs_shape, hidden_shape, action_shape, state_shape,size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            transitions,
            loss_prior=[0.]
    ):
        """Store experience and priority."""
        # previous data would be deleted according to priorities
        ptr = self._delete_proportional(loss_prior)[0]
 
        self.obs_buf[ptr] =th.tensor( transitions["agent_obs"]).detach().squeeze().cpu().numpy()
        self.hidden_buf[ptr] = th.stack( transitions["hidden_state"]).detach().squeeze().cpu().numpy()

        self.avail_buf[ptr] =th.tensor( transitions["avail_actions"]).squeeze().cpu().numpy()

        self.state_buf[ptr] = th.tensor(transitions["state"]).squeeze().cpu().numpy()
        self.pre_state_buf[ptr] = th.tensor(transitions["pre_state"]).squeeze().cpu().numpy()

        self.pre_avail_buf[ptr] = th.tensor(transitions["pre_avail_actions"]).detach().squeeze().cpu().numpy()
        self.pre_obs_buf[ptr] =th.tensor( transitions["pre_agent_obs"]).detach().squeeze().cpu().numpy()
        self.pre_hidden_buf[ptr] = th.stack( transitions["pre_hidden_state"]).detach().squeeze().cpu().numpy()

        self.pre_acts_buf[ptr] = (transitions["pre_agent_actions"].reshape((3, 1))).detach().cpu().numpy()
        self.reward_buf[ptr] = transitions["reward"].cpu().numpy()
        self.imitate_buf[ptr] = transitions["imitate_reward"]


        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, beta: float = 0.6) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        hidden = self.hidden_buf[indices]
        state = self.state_buf[indices]

        reward=self.reward_buf[indices]
        imitate_reward = self.imitate_buf[indices]

        pre_state = self.pre_state_buf[indices]
        pre_avail_actions=self.pre_avail_buf[indices]
        pre_agent_obs=self.pre_obs_buf[indices]
        pre_hidden_state=self.pre_hidden_buf[indices]
        pre_agent_actions=self.pre_acts_buf[indices]
        avail_actions=self.avail_buf[indices]
        imitate_reward =self.imitate_buf[indices]


        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return {
                "state": state,          
                "reward": reward,
                "imitate_reward": imitate_reward,

                "pre_state": pre_state,
                "pre_avail_actions": pre_avail_actions,
                "pre_agent_obs": pre_agent_obs,
                "pre_hidden_state": pre_hidden_state,
                "pre_agent_actions": pre_agent_actions,
                "avail_actions": avail_actions,
                "agent_obs": obs,
                "hidden_state": hidden,

                "weights": weights,
                "indices": indices}

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)
        if self.pre_acts_buf.shape[0] >= self.max_size:
            for idx, priority in zip(indices, priorities):
                assert priority > 0
                assert 0 <= idx < len(self)
                self.sum_tree[idx] = priority ** self.alpha
                self.min_tree[idx] = priority ** self.alpha
                self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _delete_proportional(self, loss_prior) -> List[int]:
        """Sample indices based on proportions reversely."""
        indices = []

        for prior in loss_prior:
            idx = self.sum_tree.i_retrieve(self.max_size)
            self.sum_tree[idx] = self.max_priority ** self.alpha + prior
            self.min_tree[idx] = self.max_priority ** self.alpha + prior
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
