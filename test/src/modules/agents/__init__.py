REGISTRY = {}

from .rnn_agent import RNNAgent

from .latent_ce_dis_rnn_agent import LatentCEDisRNNAgent
from .oppnent import EnemyAgent, EnemiesAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_ce_dis_rnn"] = LatentCEDisRNNAgent
REGISTRY["enemy"] = EnemyAgent
REGISTRY["enemies"] = EnemiesAgent

