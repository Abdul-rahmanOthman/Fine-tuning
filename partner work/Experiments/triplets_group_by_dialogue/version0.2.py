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
from train_embed_disease_group import dataloader as embedding_loader
from train_embed_disease_group import TripletModelTrainer


trainer = TripletModelTrainer()
trainer.train(embedding_loader, num_epochs=1000)
emb_model = trainer.model
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

gym.envs.register(
    id='ConsultationEnv-v1',
    entry_point='__main__:ConsultationEnv',
)

env_working = gym.make('ConsultationEnv-v1', symptoms=d_.symptoms_list, diseases= d_.disease_list, dataClass = d_.dataClass_train)

gym.envs.register(
    id='embedding-v1',
    entry_point='__main__:embEnv',
)
gym.envs.register(
    id='embeddingtest-v1',
    entry_point='__main__:embEnv',
)

env_notworking = gym.make('embedding-v1', symptoms=d_.symptoms_list, diseases= d_.disease_list, model = emb_model)
env_notworking_test = gym.make('embeddingtest-v1', symptoms=d_.symptoms_list, diseases= d_.disease_list, model = emb_model)


#%% Example test case
env_working.reset()
for _ in range(10):
    action = env_working.action_space.sample()  # Random action
    state, reward, done, truncated,  _ = env_working.step(action)
    print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
    if done:
        break


#%% Example test case
env_notworking.reset()
for _ in range(10):
    action = env_notworking.action_space.sample()  # Random action
    state, reward, done, truncated, _ = env_notworking.step(action)
    print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
    assert env_notworking.observation_space.contains(state), "The state must be within the observation space"
    assert 0 <= action < env_notworking.action_space.n, "Action must be within the valid range"

    if done:
        break

from stable_baselines3.common.env_checker import check_env
check_env(env_notworking)

#%%
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
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


#%%
def learn(best_performing_params):
    
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

    model.learn(total_timesteps=100000, progress_bar=True)
    epochs_hits = 0
    for _ in range(d_.length_test):
        done = False
        obs, _ = env_notworking_test.reset()
        
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env_notworking_test.step(action)
            if env_notworking_test.render():
                epochs_hits += 1

    print("Accuracy: ", epochs_hits/d_.length_test)
    print("Hits ", epochs_hits)
    print("Total ", d_.length_test)

#%%


from tqdm import tqdm


#%%
import optuna

def objective(trial):
    optuna_config = {
    'learning_rate': trial.suggest_float('alpha', 0.0001, 0.01, log = True),
    'gamma': trial.suggest_float('gamma', 0.9, 0.99, log = True),
    'vf_coef': trial.suggest_float('vf_coeff', 0.01, 1.0, log = True),
    'ent_coef': trial.suggest_float('entropy_coefficient', 0.00001, 0.01, log = True),
    'clip_range': trial.suggest_float('policy_clip', 0.01, 0.3, log = True),
    'gae_lambda': trial.suggest_float('gae_lambda', 0.7, 0.9999, log = True),
    'batch_size': trial.suggest_int('batch_size', 64, 256),
    'n_steps': trial.suggest_int('N_steps', 128, 512),
    }
    learn(optuna_config)



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)
# %%
