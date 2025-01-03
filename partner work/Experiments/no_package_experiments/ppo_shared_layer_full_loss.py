'''
THIS CODE RUNs A SHARED LAYER BUT COMBINES THE LOSS OF POLICY AND VALUE FUNCTION

THE CODE RUNS WITH MINIBATCH SIZE OF 128, 40 TRAINING ITERATIONS FOR BOTH POLICY AND VALUE FUNCTION

THE CODE RUNS FOR 10 EPOCHS

THERE IS NO DISEASE CLASSIFIER IMPLEMENTED IN THIS EXPERIMENT



'''
DEVICE = 'cuda'
import random
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import json
import copy
import numpy as np
import pandas as pd

# MD or DXY
use_data = 'DX'

if use_data == 'MD':
  dataset = pd.read_pickle(r"C:\Users\\7mood\Desktop\RLProject\Data\train.pk")
  
  with open(r"C:\Users\\7mood\Desktop\RLProject\Data\symptom.txt") as file:
    symptoms_list= []
    for line in file:
      line = line.strip()
      symptoms_list.append(line)

  with open(r"C:\Users\\7mood\Desktop\RLProject\Data\disease.txt") as file:
    disease_list = []
    for line in file:
      line = line.strip()
      disease_list.append(line)
else:
  with open(r"C:\\Users\\7mood\\Desktop\\RLProject\\MedicalDiagnosis\\dxy_data.txt") as file:
    dataset = json.load(file)

  with open(r"C:\\Users\\7mood\\Desktop\\RLProject\\MedicalDiagnosis\\symptoms_list.txt") as file:
    symptoms_list= []
    for line in file:
      line = line.strip()
      symptoms_list.append(line)

  with open(r"C:\\Users\\7mood\\Desktop\\RLProject\\MedicalDiagnosis\\disease_list.txt") as file:
    disease_list= []
    for line in file:
      line = line.strip()
      disease_list.append(line)
updated_symptom_list = symptoms_list 

num_of_symptoms = len(symptoms_list)
num_of_diseases = len(disease_list)

action_list = symptoms_list + disease_list
#%% User Simulator Class
class user():
  def __init__(self, symptoms_list, action_list, goal_list, disease_list, input_dim = None, positive_reward = 10, negative_reward = -10, asking_reward = -1):
    self.input_dim = input_dim
    self.state = {}
    self.action_list = copy.deepcopy(action_list) # actions the user will take(inform symptom status, request disease, close)
    self.symptoms_list = copy.deepcopy(symptoms_list)# all the symptoms
    self.user_goals = copy.deepcopy(goal_list)
    self.disease_list = copy.deepcopy(disease_list)
    self.max_turn = 20
    self.positive_reward = positive_reward
    self.negative_reward = negative_reward
    self.asking_reward = asking_reward


  def _reset_(self):
    self.state_vector = torch.zeros(self.input_dim, dtype = torch.float)
    self.reward = 0
    self.slots_to_index = {slot: i for i, slot in enumerate(self.symptoms_list)}
    self.disease_to_index = {disease: i for i, disease in enumerate(self.disease_list)}
    self.current_goal = get_example()
    self.implicit_symptoms = copy.deepcopy(self.current_goal['implicit_inform_slots'])
    self.disease = copy.deepcopy(self.current_goal['disease_tag'].lower())
    self.turn = 0
    self.dialogue_status = None
    self.episode_over = False

    self.history_slots = list(self.current_goal['explicit_inform_slots'].keys())
    #initialize interior state
    self.state = {}
    self.state['disease'] = 'UNK'
    self.state['user_inform'] = copy.deepcopy(self.current_goal['explicit_inform_slots'])
    self.state['action'] = 'request'


    response = {}
    response['disease'] = copy.deepcopy(self.state['disease'])
    response['user_inform'] = copy.deepcopy(self.state['user_inform'])
    response['action'] = copy.deepcopy(self.state['action'])

    self.update_state(response)
    

    return copy.deepcopy(self.state_vector), copy.deepcopy(self.current_goal)
    

  def _step_(self, agent_action):
    self.hit = 0
    self.turn += 1
    action = agent_action['action']

    if self.turn >= self.max_turn:
      self.dialogue_status = False
      self.episode_over = True
    else:
      if action == "inform":
        self.response_inform_disease(agent_action)
      elif action == "request":
        self.response_request_symptoms(agent_action)
      # elif action == "closing":
      #   self.response_closing(agent_action)

    
    response = {}
    response['disease'] = copy.deepcopy(self.state['disease'])
    response['user_inform'] = copy.deepcopy(self.state['user_inform'])
    response['action'] = copy.deepcopy(self.state['action'])

    self.update_state(response)

    
    self.state['user_inform'] = {}

    temp_reward = copy.deepcopy(self.reward)

    self.reward = 0

    return copy.deepcopy(self.state_vector), temp_reward , copy.deepcopy(self.dialogue_status), \
    copy.deepcopy(self.episode_over) # I should return a reward
  
  #### INFORM DISEASE
  def response_inform_disease(self, agent_action):
    slot = agent_action['disease'].lower()
    assert (slot not in self.symptoms_list), "Slot found in symptoms set"


    self.episode_over = True
    
    if slot != self.disease:
      self.dialogue_status = False
      self.reward = self.negative_reward
    else:
      self.dialogue_status = True
      self.reward = self.positive_reward
    
    self.state['disease'] = slot

  #### REQUEST SYMPTOM
  def response_request_symptoms(self, agent_action):

    self.state['action'] = 'inform'
    #print("The agent action is : ", agent_action)

    slot = list(agent_action['request_slots'].keys())[0]

      

    if(slot not in self.history_slots):
      self.hit += 1
      if(slot in list(self.current_goal['implicit_inform_slots'].keys())):
        if self.current_goal['implicit_inform_slots'][slot] == True: #True == 1, False == 0 not-sure = -1
          self.state['user_inform'][slot] = 3.0 # present
         # self.reward+= 0.5
        elif self.current_goal['implicit_inform_slots'][slot] == False:
          self.state['user_inform'][slot] = 2.0 # absent
         # self.reward+= 0.2
      else:
       # self.reward -= 0.1
        
        self.state['user_inform'][slot] = 1.0
      
      self.history_slots.append(slot)

    else: pass
    # self.reward-= 0.15

    self.reward = self.asking_reward
  
  def update_state(self, slots):
    '''
    slots -> 
    {
    'disease' : UNK or 'disease_name'
    'user_inform' : {'symptom' : value, ...}
    'action': user either request or inform
    }
    '''
    disease = slots['disease']
    symptoms = slots['user_inform']

    for slot, value in symptoms.items():
      if slot in self.slots_to_index:
        self.state_vector[self.slots_to_index[slot]] = value # this updates the state vector in shape of [2,3,1,0,0,1,2,1,1,3 ...]

    #additional disease slot can be made here

"""---

### HERE STARTS AGENT CODE
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
  def __init__(self, obs_space_size, action_space_size):
    super().__init__()

    self.shared_layers = nn.Sequential(
      nn.Linear(obs_space_size, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU())

    self.policy_layers = nn.Sequential(
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, action_space_size),
      nn.Softmax(dim=-1)
      )

    self.value_layers = nn.Sequential(
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 1))

  def value(self, state):
    z = self.shared_layers(state)
    value = self.value_layers(z)
    return value
  
  def policy(self, state):
    z = self.shared_layers(state)
    policy_logits = self.policy_layers(z)
    return policy_logits
  
  def forward(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    dist = Categorical(policy_logits)
    value = self.value_layers(z)
    
    return dist, value

"""---

###TRAINER CLASS
"""

class Trainer():
  def __init__(self,
              actor_critic,
              ppo_clip_val=0.2,
              target_kl_div=0.01,
              max_policy_train_iters=80,
              value_train_iters=80,
              policy_lr=3e-4,
              value_lr=1e-2,
              vf_coeff = 0.5,
              entropy_coeff = 0.01,
              symptoms_list= None,
              disease_list= None,
              action_set = None,
              input_dim = None):
    self.turn = 0
    self.vf_coeff = vf_coeff
    self.ac = actor_critic
    self.ppo_clip_val = ppo_clip_val
    self.target_kl_div = target_kl_div # i dont know, some hyperparameter
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters
    self.entropy_coeff = entropy_coeff
    self.symptoms_list = symptoms_list
    self.disease_list = disease_list
    self.action_set = action_set
    self.symp_size = len(symptoms_list)
    self.input_dim = input_dim



    policy_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.policy_layers.parameters())
    self.policy_optim = optim.Adam(policy_params, lr=policy_lr)

    value_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.value_layers.parameters())
    self.value_optim = optim.Adam(value_params, lr=value_lr)

  def train_policy_value(self, obs, acts, old_log_probs, gaes, returns, entropy, minibatch_size):
      avg_policy_loss = 0.0
      avg_value_loss = 0.0
      num_batches = len(obs) // minibatch_size

      for _ in range(self.max_policy_train_iters):
          permute_idxs = np.random.permutation(len(obs))
          for i in range(num_batches):
              idxs = permute_idxs[i * minibatch_size: (i + 1) * minibatch_size]

              batch_obs = obs[idxs]
              batch_acts = acts[idxs]
              batch_old_log_probs = old_log_probs[idxs]
              batch_gaes = gaes[idxs]
              batch_returns = returns[idxs]
              batch_entropy = entropy[idxs]

              self.policy_optim.zero_grad()
              self.value_optim.zero_grad()

              # Policy loss calculation
              new_logits = self.ac.policy(batch_obs)
              new_logits = Categorical(logits=new_logits)
              new_log_probs = new_logits.log_prob(batch_acts)
              assert (new_log_probs.cpu().sum() != 1), f"{new_log_probs.cpu().sum()}, new log probs is not 1"

              policy_ratio = torch.exp(new_log_probs - batch_old_log_probs)
              clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)

              entropy_bonus = -self.entropy_coeff * batch_entropy.mean()
              clipped_loss = clipped_ratio * batch_gaes
              full_loss = policy_ratio * batch_gaes
              policy_loss = -torch.min(full_loss, clipped_loss).mean() + entropy_bonus
              avg_policy_loss += policy_loss

              # Value loss calculation
              values = self.ac.value(batch_obs)
              value_loss = (batch_returns - values) ** 2
              value_loss = value_loss.mean()
              avg_value_loss += value_loss

              # Combined loss
              combined_loss = policy_loss + self.vf_coeff * value_loss
              combined_loss.backward()

              self.policy_optim.step()
              self.value_optim.step()

              kl_div = (batch_old_log_probs - new_log_probs).mean()
              if kl_div >= self.target_kl_div:
                  break

        #print('avg policy loss on train iters = ', avg_policy_loss / (self.max_policy_train_iters * num_batches))
        #print('avg value loss on train iters = ', avg_value_loss / (self.value_train_iters * num_batches))


  def send_action(self, action):
      #print("action = ", action)
      if action_list[action] in symptoms_list:
        return {'action': 'request', 'request_slots': {action_list[action]: "" }, 'disease': 'UNK'}
      else:
        return {'action': 'inform', 'request_slots': {},'disease': action_list[action]}
      
  



DEVICE = 'cuda'
class Manager():
  def __init__(self, dataset= None, action_list =None ,symptoms_list=None, disease_list=None, input_dim = None, config=None ):
    self.input_dim = input_dim
    self.dataset = dataset
    self.action_list = action_list
    self.symptoms_list = symptoms_list
    self.disease_list = disease_list
    # hyperparameters for user
    positive_r= config['positive_reward']
    negative_r = config['negative_reward']
    asking_r = config['asking_reward']
    self.user = user(symptoms_list, action_list, dataset, disease_list, input_dim=self.input_dim, positive_reward=positive_r, negative_reward=negative_r, asking_reward=asking_r)

    # Hyperparameters for training
    self.minibatch_size= config['batch_size']
    self.gamma = config['gamma']
    self.gae_lambda = config['gae_lambda']
    self.decay = config['decay']
    self.max_turns = 20
    # Hyperpraraters for PPO agent
    self.ppo_clip = config['policy_clip']
    self.ent_coeff = config['entropy_coefficient']
    policy_lr = config['policy_lr']
    value_lr = config['value_lr']
    self.n_steps = config['N_steps']
    self.vf_coeff = config['vf_coeff']

    self.train_episodes = int(0.8*len(dataset))
    self.test_episodes = len(dataset) - self.train_episodes

    self.agent = ActorCriticNetwork(obs_space_size=input_dim, \
                                    action_space_size= len(action_list))

    self.trainer = Trainer( self.agent, policy_lr = policy_lr, value_lr = value_lr, target_kl_div = 0.02, \
                           max_policy_train_iters = 40, value_train_iters = 40, symptoms_list= symptoms_list,\
                            action_set= action_list, disease_list=disease_list, input_dim= self.input_dim, vf_coeff = self.vf_coeff,entropy_coeff = self.ent_coeff, ppo_clip_val = self.ppo_clip)


  
  
  
  def train_loop(self):
    self.hit = 0
    ep_rewards = []
    accumulated_data = [[], [], [], [], [], []] #obs, act, reward, val, act_log_prob, entropy

    for episode_idx in range(self.train_episodes):

        train_data, reward = self.rollout(self.agent, self.user)
        ep_rewards.append(reward)
        
        
        for i in range(len(train_data)):
            accumulated_data[i].extend(train_data[i])

        
        if (episode_idx + 1) % self.n_steps == 0 or (episode_idx + 1) == self.train_episodes:
            # Shuffle data
            permute_idxs = np.random.permutation(len(accumulated_data[0]))
            
            # Prepare training data
            obs = torch.tensor([accumulated_data[0][i] for i in permute_idxs],
                               dtype=torch.float32, device=DEVICE)
            acts = torch.tensor([accumulated_data[1][i] for i in permute_idxs],
                                dtype=torch.int64, device=DEVICE)
            gaes = torch.tensor([accumulated_data[3][i] for i in permute_idxs],
                                dtype=torch.float32, device=DEVICE)
            act_log_probs = torch.tensor([accumulated_data[4][i] for i in permute_idxs],
                                         dtype=torch.float32, device=DEVICE)
            entropy = torch.tensor([accumulated_data[5][i] for i in permute_idxs],
                                   dtype=torch.float32, device=DEVICE)
            
           
            returns = discount_rewards([accumulated_data[2][i] for i in permute_idxs], gamma = self.gamma)
            returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

            
            self.trainer.train_policy_value(obs, acts, act_log_probs, gaes, returns, entropy, self.minibatch_size)

            
            accumulated_data = [[], [], [], [], [], []]

    return self.hit
  
  def rollout(self, model, user):
    """
    Performs a single rollout.
    Returns training data in the shape (n_turns, observation_shape)
    and the cumulative reward.
    """
    ### Create data storage
    train_data = [[], [], [], [], [], []] # obs, act, reward, values, act_log_probs, entropy
    turn = 0
    obs, example  = user._reset_()
    
    ep_reward = 0

    for _ in range(self.max_turns):
      
      turn += 1
      dist, val = model(torch.tensor(obs, dtype=torch.float32,
                                        device=DEVICE).unsqueeze(0))
 
      
      
      act = dist.sample()
      act_log_prob = dist.log_prob(act).item()      
      entropy = dist.entropy().item()

      act, val = act.item(), val.item()

      encode_act = self.trainer.send_action(act)
      
      next_obs, reward , dialogue_status , done  = user._step_(encode_act)
      
      for i, item in enumerate((obs, act, reward, val, act_log_prob, entropy)):
        train_data[i].append(item)

      obs = next_obs
      ep_reward += reward

      if done:
        #print(example['disease_tag'], user.state['disease'])
        if dialogue_status:
          self.hit += 1
        break
    
    # handling edgy cases
    train_data = convert_train_data(train_data)

    
    train_data[3] = calculate_gaes(train_data[2], train_data[3], gae_lambda=self.gae_lambda, decay= self.decay)
    return train_data, ep_reward

  def test_accuracy(self, model, user):
      self.hit = 0
      turn = 0
      print("test episodes ", self.test_episodes)
      done = False
      ep_reward = 0
      for _ in range(self.test_episodes): 
        obs, example  = user._reset_()
        done = False
        while not done:
          
          turn += 1
          dist, val = model(torch.tensor(obs, dtype=torch.float32,
                                            device=DEVICE).unsqueeze(0))
    
          act = dist.sample()
          act, val = act.item(), val.item()
          encode_act = self.trainer.send_action(act)
          
          next_obs, reward, dialogue_status , done  = user._step_(encode_act)

          obs = next_obs
          ep_reward += reward

        if dialogue_status:
          
          self.hit += 1
      
      print (f"Test Accuracy: {(self.hit / self.test_episodes) * 100:.2f}%")
  


def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gae_lambda=0.99, decay=0.97):
    """
    Return the GAE from the given rewards and values.
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gae_lambda * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gae_lambda * gaes[-1])

    return np.array(gaes[::-1])

def convert_train_data(train_data):
    processed_data = []
    for sublist in train_data:
        # Check if the sublist contains tensors
        if isinstance(sublist[0], torch.Tensor):
            # Convert tensors to numpy arrays and flatten them if necessary
            processed_data.append(np.vstack([x.cpu().numpy() for x in sublist]))
        else:
            # Convert integers or floats directly to a NumPy array of floats
            processed_data.append(np.array(sublist, dtype=np.float64))
    return processed_data


number_of_ex_taken = 0
Train = True
def get_example(reset = False):
  global dataset
  global number_of_ex_taken
  random.shuffle(dataset)
  train_dataset = dataset[:int(0.8 * len(dataset))]
  test_dataset = dataset[int(0.2 * len(dataset)):]
  global Train
  if Train:
    if reset:
        random.shuffle(dataset)
        number_of_ex_taken = 0  # Reset the counter
        return None
    
    example = train_dataset[number_of_ex_taken]  # Access the element
    number_of_ex_taken += 1  # Increment the counter
    return example
  
  else:
    if reset:
        random.shuffle(dataset)
        number_of_ex_taken = 0  # Reset the counter
        return None
    
    example = test_dataset[number_of_ex_taken]  # Access the element
    number_of_ex_taken += 1  # Increment the counter
    return example

import matplotlib.pyplot as plt
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


configuration = {
  'positive_reward' : 10,
  'negative_reward' : -13,
  'asking_reward' : -5,
  'policy_lr': 0.0037,
  'value_lr': 0.001,
  'gamma': 0.92,
  'vf_coeff': 0.1879601677,
  'entropy_coefficient': 0.06057,
  'policy_clip': 0.2375,
  'gae_lambda': 0.814,
  'batch_size': 128,
  'N_steps' : 256,
  'n_epochs': 10,
  'decay': 0.97
}
from tqdm import tqdm
def run_model(dataset, action_list, symptoms_list, disease_list, config):
    """
    Runs the training process and returns the accuracy.

    Args:
    - dataset: The dataset used for training.
    - action_list: List of possible actions.
    - symptoms_list: List of possible symptoms.
    - disease_list: List of possible diseases.
    - config: Configuration dictionary containing hyperparameters.
    - n_epochs: Number of training epochs.

    Returns:
    - accuracy: The final accuracy after training.
    """

    input_dims =  num_of_symptoms
    n_epochs = config['n_epochs']
    
    manager = Manager(dataset, action_list, symptoms_list, disease_list, input_dims, config=config)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    for epoch in tqdm(range(n_epochs)):
        manager.train_loop()
        print(f"Epoch {epoch + 1} hits = {manager.hit}")
        get_example(reset=True)

    get_example(reset=True)
    # Calculate accuracy
    accuracy = manager.hit / train_size
    print("Training hits ", manager.hit)
    print (f"Training Accuracy: {accuracy * 100:.2f}%")
    print("Testing...")
    global Train
    Train = False
    print("Test Accuracy: ", manager.test_accuracy(manager.agent, manager.user))
    print("Test hits ", manager.hit)
    

run_model(dataset, action_list, symptoms_list, disease_list, configuration)
print("Done")