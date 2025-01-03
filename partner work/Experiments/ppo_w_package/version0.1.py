#%%
DEVICE = 'cuda'
import random
import torch
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import dataset1 as d_
from dataset1 import ConsultationEnv
from dataset1 import embEnv 
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


best_performing_params = {'learning_rate': 0.0037134961264765898, 'n_steps': 256, 'batch_size': 128, 'gamma': 0.9487210238458803, 
                          'gae_lambda': 0.8136627047202266, 'ent_coef': 0.006057819640776694, 'vf_coef': 0.18796016773643454, 
                          'clip_range': 0.23752465643218776, 'positive_reward': 13, 'negative_reward': -5, 'asking_reward': -3}



from functools import partial

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from train_embeddings import dataloader as embedding_loader
from train_embeddings import TripletModelTrainer

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

#%%

gym.envs.register(
    id='ConsultationEnv-v1',
    entry_point='__main__:ConsultationEnv',
)
gym.envs.register(
    id='ConsultationEnvtest-v1',
    entry_point='__main__:ConsultationEnv',
)

env_working = gym.make('ConsultationEnv-v1', symptoms=d_.symptoms_list, diseases= d_.disease_list, dataClass = d_.dataClass_train)
env_working_test = gym.make('ConsultationEnvtest-v1', symptoms=d_.symptoms_list, diseases= d_.disease_list, dataClass = d_.dataClass_test)





#%%
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
#

class customMLPExtractor(MlpExtractor):
    def __init__(self, *args, **kwargs):
        super(customMLPExtractor, self).__init__(*args, **kwargs)

        self.policy_net.float()
        self.value_net.float()




import torch as th
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

class policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(policy, self).__init__(*args, **kwargs,
                                     share_features_extractor=True)
        

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            # print the dtypes of the features
            

            try:
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)

            except:
                raise ValueError("custom failure in forward pass of the policy network")
        # Evaluate the values for the given observations

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob
    
    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = customMLPExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.action_net.float()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1).float()
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    
model = PPO(policy, env_working, verbose=0, 
            learning_rate=best_performing_params['learning_rate'], 
            n_steps=best_performing_params['n_steps'], 
            batch_size=best_performing_params['batch_size'], 
            gamma=best_performing_params['gamma'], 
            gae_lambda=best_performing_params['gae_lambda'], 
            ent_coef=best_performing_params['ent_coef'], 
            vf_coef=best_performing_params['vf_coef'], 
            clip_range=best_performing_params['clip_range'],
            device = 'cuda')

# Train the agent

model.learn(total_timesteps=100000, progress_bar=True)
#%%


from tqdm import tqdm

epochs_hits = 0

for _ in range(d_.length_test):
    done = False
    obs, _ = env_working_test.reset()
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env_working_test.step(action)
        if env_working_test.render():
            epochs_hits += 1

print(f"Tested on {d_.length_test} episodes, hit the target in {epochs_hits} episodes")
print(f"Success rate: {epochs_hits / d_.length_test:.2f}")

