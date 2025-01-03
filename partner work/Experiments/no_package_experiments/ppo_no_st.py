''' user goal
{'request_slots': {'disease': 'UNK'},
 'implicit_inform_slots': \{'sneeze': False, 'allergy': True},
 'explicit_inform_slots': {'cough': True, 'runny nose': True},
 'disease_tag': 'allergic rhinitis',
 'consult_id': 'train-0'}

'''
'''
Following methodolgies are included here:
1- state_vector is used to represent the state of the user (size of # of symptoms)
2- Tanh can be used but needs a modification in the state_vector to one hot encoded vector
3-  Adam optimizer is used for the actor and critic networks
4-  input_dims = num_of_symptoms

'''
#%% Import Data and set library
DEVICE = 'cuda'
import random
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import json
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
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
    self.input_dim = num_of_symptoms
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

    self.current_goal = get_example()
    self.implicit_symptoms = copy.deepcopy(self.current_goal['implicit_inform_slots'])
    self.disease = copy.deepcopy(self.current_goal['disease_tag'])
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
    
    self.state['user_inform'] = {}
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

    
    response = {}
    response['disease'] = copy.deepcopy(self.state['disease'])
    response['user_inform'] = copy.deepcopy(self.state['user_inform'])
    response['action'] = copy.deepcopy(self.state['action'])

    self.update_state(response)

    self.state['user_inform'] ={}
    temp_reward = copy.deepcopy(self.reward)

    self.reward = 0

    return copy.deepcopy(self.state_vector), temp_reward , copy.deepcopy(self.dialogue_status), \
    copy.deepcopy(self.episode_over) # I should return a reward
  
  #### INFORM DISEASE
  def response_inform_disease(self, agent_action):
    slot = agent_action['disease']
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

    #print('slot = ', slot)
    # check condition on the network index
    #assert (slot not in self.disease_list), "Wrong slot taken"
      

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

    for slot, value in slots['user_inform'].items():
      if slot in self.slots_to_index:
        self.state_vector[self.slots_to_index[slot]] = value # this updates the state vector in shape of [2,3,1,0,0,1,2,1,1,3 ...]

    #additional disease slot can be made here

#%% Memory Class
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        if n_states % self.batch_size != 0:
          batches.append(indices[:n_states % self.batch_size])

        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = [] 
        self.vals = []

from tqdm import tqdm
class Critic(nn.Module):
    #  alpha, fc1_dims=512, fc2_dims=512
    def __init__(self, input_dims, alpha, fc1_dims=64, fc2_dims=64):
      super(Critic, self).__init__()
      self.critic = nn.Sequential(
              nn.Linear(input_dims, fc1_dims),
              nn.ReLU(),
              nn.Linear(fc1_dims, 1)
      )
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.to(self.device)
      self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
      value = self.critic(state)
      return value
#%% Actor class
class Actor(nn.Module):
  #  alpha, fc1_dims=512, fc2_dims=512
  def __init__(self, n_actions, input_dims, alpha, fc1_dims=64, fc2_dims=64):
    super(Actor, self).__init__()

    self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)
    self.optimizer = optim.Adam(self.parameters(), lr=alpha)
  # FORWARD function
  def forward(self, state):
    dist = self.actor(state)
    dist = Categorical(dist)
    return dist
#%% Agent Class (networks and memory, take states return actions)
class Agent:
  def __init__(self, n_actions, input_dims, gamma=0.95, alpha=0.0037, gae_lambda=0.814,
        policy_clip=0.2375, batch_size=128, symptoms_list = None, disease_list = None, vf_coeff = 0.1879601677, ent_coeff = 0.006057):
    
    self.gamma = gamma
    self.policy_clip = policy_clip
    self.gae_lambda = gae_lambda
    self.vf_coeff = vf_coeff
    self.ent_coeff = ent_coeff

    self.actor = Actor(n_actions, input_dims, alpha)
    self.critic = Critic(input_dims, alpha)
    self.memory = PPOMemory(batch_size)
    self.symptoms_list = symptoms_list
    self.disease_list = disease_list

  def remember(self, state, action, probs, vals, reward, done):
    self.memory.store_memory(state, action, probs, vals, reward, done)

  def choose_action(self, observation):
    dist = self.actor(observation)
    
    value = self.critic(observation)
    action = dist.sample()

    probs = torch.squeeze(dist.log_prob(action)).item()
    action = torch.squeeze(action).item()
    value = torch.squeeze(value).item()

    act_as_dict = self.wrap_action(action_list[action])

    return action, act_as_dict, probs, value

  def expert_action(self, observation):
      
      probs = 1.0  # Since this is the expert action, you can assume it has the highest probability
      value = self.critic(observation)
      
      value = torch.squeeze(value).item()
      #print("value = ", value, "in expert_action")
      if len(self.implicit_slots) == 0:
          action = self.slot_to_index(self.current_dis) 
          act_as_dict = self.wrap_action(action_list[action])
      else: 
          action_as_slot = random.choice(self.implicit_slots)
          self.implicit_slots.remove(action_as_slot)
          action = self.slot_to_index(action_as_slot)
          act_as_dict = self.wrap_action(action_as_slot)
      
      return action, act_as_dict , probs, value

  def slot_to_index(self, slot):
      return action_list.index(slot)

  def wrap_action(self, action):
      #print("action = ", action)
      if action in symptoms_list:
        return {'action': 'request', 'request_slots': {action: "" }, 'disease': 'UNK'}
      else:
        return {'action': 'inform', 'request_slots': {},'disease': action}
      
    
  def initialize_episode(self, example):
     self.implicit_slots = list(example['implicit_inform_slots'].keys())
     self.current_dis = example['disease_tag']

  def learn(self):
      
          
      state_arr, action_arr, old_prob_arr, vals_arr,\
      reward_arr, dones_arr, batches = \
        self.memory.generate_batches()
      


      values = vals_arr
      advantage = np.zeros(len(reward_arr), dtype=np.float32)

      for t in range(len(reward_arr)-1):
          discount = 1
          a_t = 0
          for k in range(t, len(reward_arr)-1):
              a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                      (1-int(dones_arr[k])) - values[k])
              discount *= self.gamma*self.gae_lambda
          advantage[t] = a_t
      
      advantage = torch.tensor(advantage).to(self.actor.device)
      values = torch.tensor(values).to(self.actor.device)
      for batch in batches:
          #print("batch in batches = ", batch)
          states = torch.vstack([state_arr[i] for i in batch]).float().to(self.actor.device)
          #print("states in batch = ", len(states), states)
          old_probs = torch.tensor([old_prob_arr[i] for i in batch]).to(self.actor.device)
          actions = torch.tensor([action_arr[i] for i in batch], dtype=torch.int64).to(self.actor.device)

          dist = self.actor(states)
          critic_value = self.critic(states)

          critic_value = torch.squeeze(critic_value)

          new_probs = dist.log_prob(actions)
          prob_ratio = new_probs.exp() / old_probs.exp()
          #prob_ratio = (new_probs - old_probs).exp()
          weighted_probs = advantage[batch] * prob_ratio
          weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                  1+self.policy_clip)*advantage[batch]
          actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

          returns = advantage[batch] + values[batch]
          critic_loss = (returns-critic_value)**2
          critic_loss = critic_loss.mean()

          # Calculate entropy
          entropy = dist.entropy().mean()

          # Total loss
          total_loss = actor_loss + self.vf_coeff * critic_loss - self.ent_coeff * entropy

          self.actor.optimizer.zero_grad()
          self.critic.optimizer.zero_grad()
          total_loss.backward()
          self.actor.optimizer.step()
          self.critic.optimizer.step()

      self.memory.clear_memory()


class Manager: 
  def __init__(self, dataset= None, action_list =None ,symptoms_list=None, disease_list=None, input_dim = None, config=None ):
    self.dataset = dataset
    self.config = config
    self.action_list = action_list
    self.symptoms_list = symptoms_list
    self.disease_list = disease_list
    self.input_dim = input_dim
    self.action_space_size = len(action_list)
    self.supervised = False # 1 for expert action, 0 for agent action


    positive_r= config['positive_reward']
    negative_r = config['negative_reward']
    asking_r = config['asking_reward']
    self.user = user(symptoms_list, action_list, disease_list, input_dim, positive_r, negative_r, asking_r)
    

    alpha = config['alpha']
    batch_size = config['batch_size']
    self.N = config['N_steps']
    ppo_clip = config['policy_clip']
    gamma = config['gamma']
    gae_lambda = config['gae_lambda']
    vf_coeff = config['vf_coeff']
    ent_coeff = config['entropy_coefficient']
    self.agent = Agent(n_actions=self.action_space_size, batch_size=batch_size, 
               alpha=alpha, gamma=gamma, gae_lambda=gae_lambda, policy_clip=ppo_clip,
               input_dims=self.input_dim, vf_coeff=vf_coeff, ent_coeff=ent_coeff)
    
    self.train_episodes = int(0.8*len(dataset))
    self.test_episodes = len(dataset) - self.train_episodes
    self.n_episodes = len(dataset)
    self.num_expert_eps = 50
    self.hit = 0
    
  def run_episodes(self):
    self.hit = 0
    learn_iters = 0
    figure_file = 'plots/cartpole.png'

    best_score = -np.inf
    score_history = []
    
    n_steps = 0
    avg_score = 0
    for i in range(self.train_episodes):
      done = False
      score = 0
      observation, example = self.user._reset_()

      self.agent.initialize_episode(example)
      score = 0
      if (i+1) % self.num_expert_eps  == 0:
        self.supervised = False

      while not done:
          n_steps += 1
          if self.supervised:
            action, act_as_dict ,prob, val = self.agent.expert_action(observation)

          else:
            action, act_as_dict, prob, val = self.agent.choose_action(observation)


          observation_, reward, dialogue_status, done = self.user._step_(act_as_dict)

          score += reward
          
          self.agent.remember(observation_, action, prob, val, reward, done)
        
          if n_steps % self.N == 0:
              self.agent.learn()
            
        
          observation = observation_
    
      if dialogue_status:
        self.hit += 1


      score_history.append(score)
      avg_score = np.mean(score_history[-100:])

      if avg_score > best_score:
          best_score = avg_score
  def test_accuracy(self, n_episodes=100):

    self.hit = 0
    
    for _ in range(self.test_episodes):
      done = False
      score = 0
      observation, example = self.user._reset_()
      self.agent.initialize_episode(example)
      n_steps = 0
      while not done:
        action, act_as_dict, prob, val = self.agent.choose_action(observation)
        observation_, reward, dialogue_status, done = self.user._step_(act_as_dict)
        score += reward
  
        observation = observation_

      if dialogue_status:
        self.hit += 1

    return self.hit / self.test_episodes


#%% Helper Functions
number_of_ex_taken = 0  
def get_example(reset = False):
  global number_of_ex_taken  # Correct use of global to modify the variable
  global dataset
  if reset:
      random.shuffle(dataset)
      number_of_ex_taken = 0  # Reset the counter
      return None
  
  example = dataset[number_of_ex_taken]  # Access the element
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
  'negative_reward' : -10,
  'asking_reward' : -1,
  'alpha': 0.0037,
  'gamma': 0.95,
  'vf_coeff': 0.1879601677,
  'entropy_coefficient': 0.006057,
  'policy_clip': 0.2375,
  'gae_lambda': 0.814,
  'batch_size': 128,
  'N_steps' : 256,
  'n_epochs': 15

}

#%% Main Function
def run_training(dataset, action_list, symptoms_list, disease_list, config):
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

    for epoch in tqdm(range(n_epochs)):
        manager.run_episodes()
        print(f"Epoch {epoch + 1} hits = {manager.hit}")
        get_example(reset=True)

    get_example(reset=True)
    # Calculate accuracy
    accuracy = manager.hit / len(dataset)
    print (f"Training Accuracy: {accuracy * 100:.2f}%")
    print("Testing...")
    global Train
    Train = False
    print("Test Accuracy: ", manager.test_accuracy(int(0.2 * len(dataset))))
    return accuracy

import optuna

def objective(trial):
  get_example(reset=True)
  optuna_config = {
    'positive_reward': trial.suggest_int('positive_reward', 10, 30),
    'negative_reward': trial.suggest_int('negative_reward', -30, -10),
    'asking_reward': trial.suggest_int('asking_reward', -5, 5),
    'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log = True),
    'gamma': trial.suggest_float('gamma', 0.9, 0.99, log = True),
    'vf_coeff': trial.suggest_float('vf_coeff', 0.01, 1.0, log = True),
    'entropy_coefficient': trial.suggest_float('entropy_coefficient', 0.00001, 0.01, log = True),
    'policy_clip': trial.suggest_float('policy_clip', 0.01, 0.3, log = True),
    'gae_lambda': trial.suggest_float('gae_lambda', 0.7, 0.9999, log = True),
    'batch_size': trial.suggest_int('batch_size', 64, 256),
    'N_steps': trial.suggest_int('N_steps', 128, 512),
    'n_epochs': 5}
    
   
  accuracy = run_training(dataset, action_list, symptoms_list, disease_list, optuna_config)
  
  return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)