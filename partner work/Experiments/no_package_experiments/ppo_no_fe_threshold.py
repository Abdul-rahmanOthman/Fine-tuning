'''
THE CODE CONSISTS OF THE FOLLOWING CLASSES:
1-  user: The user simulator class
2-  StateTracker: The state tracker class
3-  PPOMemory: The memory class for the PPO algorithm
4-  Critic: The critic class for the PPO algorithm
5-  Actor: The actor class for the PPO algorithm
6-  Agent: The agent class that uses the actor and critic classes
7-  Manager: The manager class that uses the agent class to run the episodes

Following methodolgies are included here:
1-  State Tracker: The state tracker class is used to keep track of the state of the dialogue system

2-  Adam optimizer is used for the actor and critic networks
3-  NO Edits are made to the agent action
4-  input_dims = 3* num_of_symptoms + 3
5-  DX # change later
6- The reward is set to 10 for positive, -10 for negative, -1 for asking 
  + a reward for asking relevant non redundant questions
7- No Feature enginnering is done
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




#%%


with open(r"C:\\Users\\7mood\\Desktop\\RLProject\\MedicalDiagnosis\\symptoms_list.txt") as file:
  symptoms_list= []
  for line in file:
    line = line.strip()
    symptoms_list.append(line)

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

    self.current_goal = get_example()
    self.implicit_symptoms = copy.deepcopy(self.current_goal['implicit_inform_slots'])
    self.disease = copy.deepcopy(self.current_goal['disease_tag'])
    self.state['disease'] = 'UNK'
    self.state['user_inform'] = copy.deepcopy(self.current_goal['explicit_inform_slots'])

    #update the user_inform slots from True and False to 3.0 and 2.0
    for key, value in self.state['user_inform'].items():
      if value == True:
        self.state['user_inform'][key] = 3.0
      else:
        self.state['user_inform'][key] = 2.0

    self.history_slots = list(self.current_goal['explicit_inform_slots'].keys())
    self.state['action'] = 'request'
    self.reward = 0

    response = {}
    response['disease'] = copy.deepcopy(self.state['disease'])
    response['inform_slots'] = copy.deepcopy(self.state['user_inform'])
    response['action'] = copy.deepcopy(self.state['action'])
    
    self.state['user_inform'] = {}

    self.dialogue_status = None
    self.turn = 0
    self.episode_over = False


    return copy.deepcopy(response), copy.deepcopy(self.current_goal)


  def _step_(self, agent_action):
    self.turn += 1
    action = agent_action['action']

    if self.turn >= self.max_turn:
      self.dialoge_status = False
      self.episode_over = True
    else:
      if action == "inform":
        self.response_inform_disease(agent_action)
      elif action == "request":
        self.response_request_symptoms(agent_action)


    
    response = {}
    response['disease'] = copy.deepcopy(self.state['disease'])
    response['inform_slots'] = copy.deepcopy(self.state['user_inform'])
    response['action'] = copy.deepcopy(self.state['action'])

    temp_reward = copy.deepcopy(self.reward)

    self.reward = 0

    return copy.deepcopy(response), temp_reward , copy.deepcopy(self.dialogue_status), \
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
    
    self.state['disease'] = self.disease

  #### REQUEST SYMPTOM
  def response_request_symptoms(self, agent_action):

    self.state['action'] = 'inform'
    slot = list(agent_action['request_slots'].keys())[0]
    # check condition on the network index
    self.reward = 0

    if(slot not in self.history_slots):
      if(slot in list(self.current_goal['implicit_inform_slots'].keys())):
        if self.current_goal['implicit_inform_slots'][slot] == True: #True == 1, False == 0 not-sure = -1
          self.state['user_inform'][slot] = 3.0 # present
          self.reward+= 1
        elif self.current_goal['implicit_inform_slots'][slot] == False:
          self.state['user_inform'][slot] = 2.0 # absent
          self.reward+= 1
      else:
        self.reward -= 0.1
        self.state['user_inform'][slot] = 1.0
      
      self.history_slots.append(slot)

    else: 
      self.reward -= 1

    self.reward += self.asking_reward # negative reward for spending time asking

    
  '''
  def update_state(self, slots):
    for slot, value in slots.items():
      if slot in self.slot_to_index:
      # Update the state vector at the index for this slot
          self.state_vector[self.slot_to_index[slot]] = value
  '''
#%% State tracker class
# I DID NOT ADD TURN INTO THE STATE VECTOR
class StateTracker():
  def __init__(self, symptoms_list, disease_list):
    self.symptoms_list = symptoms_list
    self.disease_list = disease_list
    self.slot_to_index = {slot: idx for idx, slot in enumerate(self.symptoms_list)}
    self.current_slots = {}
    self.history = []
    self.turn_count = 0

  def reset(self):
    self.current_slots = {}
    self.history = []
    self.turn_count = 0

    self.current_slots['inform_slots'] = {}  # all inform slots only for symptoms (all the informed slot from the beggining of the dialogue)
    self.current_slots['disease'] = "UNK" # one of 5 diseases

  """
  user_action = {'disease': 'UNK', 'inform_slots': {this is filled}, 'action': ''}
  """

  """
  agent_action = {'disease': 'filled when informing','request_slots': {this is filled}, 'action': ''}
  """
  def update(self, agent_action=None, user_action=None):
      """ Update the state based on the latest action """
      #  Make sure that the function was called properly
      assert (not (user_action and agent_action))
      assert (user_action or agent_action)
      
      #   Update state to reflect a new action by the user.
      if user_action:
      
        for key, value in user_action['inform_slots'].items(): # Add all the new inform slots by the user to the current state
          self.current_slots['inform_slots'][key] = value
          

        self.current_slots['disease'] = user_action['disease'] # Update the disease slot "initially it is UNK"

        self.history.append(user_action) # Add the user action to the history
        self.turn_count += 1

      #   Update state to reflect a new action by the agent.
      if agent_action:
        """ the agent action dictionary cosists of the following: 
        {'disease': 'filled when informing','request_slots': {this is a slot that he is asking}, 'action': ''}
        """
        self.history.append(agent_action)
        self.turn_count += 1

        if agent_action['action'] == 'inform':
          self.current_slots['disease'] = agent_action['disease'] # Update the disease slot

    
  def get_state(self):

    represent_current_slots = torch.zeros(len(self.symptoms_list), dtype=torch.float32, device=DEVICE)
    for key, value in self.current_slots['inform_slots'].items():
        represent_current_slots[self.slot_to_index[key]] = value
    
    # Determine the disease index INSTEAD of disease vector
    if self.current_slots['disease'] == "UNK":
        disease_index = torch.tensor([len(self.disease_list)], dtype=torch.long, device=DEVICE)  # "UNK" token
    else:
        try:
            disease_index = torch.tensor([self.disease_list.index(self.current_slots['disease'])], dtype=torch.long, device=DEVICE)
        except ValueError:
            disease_index = torch.tensor([self.disease_list], dtype=torch.long, device=DEVICE)  # "UNK" token

    
    if len(self.history) > 1 and 'action' in self.history[-2]:
        if self.history[-2]['action'] == 'inform':
            last_agent_action = torch.tensor([1], dtype=torch.long, device=DEVICE)
        else:
            last_agent_action = torch.tensor([2], dtype=torch.long, device=DEVICE)
    else:
        last_agent_action = torch.tensor([0], dtype=torch.long, device=DEVICE)  # none

    # Represent last user action
    if len(self.history) > 0 and 'action' in self.history[-1]:
        if self.history[-1]['action'] == 'inform':
            last_user_action = torch.tensor([1], dtype=torch.long, device=DEVICE)
        else:
            last_user_action = torch.tensor([2], dtype=torch.long, device=DEVICE)
    else:
        last_user_action = torch.tensor([0], dtype=torch.long, device=DEVICE)  # none

    # Represent user slots as a sparse vector
    represent_user_slots = torch.zeros(len(self.symptoms_list), dtype=torch.float32, device=DEVICE)
    for key, value in self.history[-1]['inform_slots'].items():
        represent_user_slots[self.slot_to_index[key]] = value

    # Represent agent slots as a sparse vector
    represent_agent_slots = torch.zeros(len(self.symptoms_list), dtype=torch.float32, device=DEVICE)
    if len(self.history) > 1:
        for key, _ in self.history[-2]['request_slots'].items():
            represent_agent_slots[self.slot_to_index[key]] = 1

    # Combine all parts of the state representation
    final_representation = torch.cat([
        represent_current_slots,  # Remove the extra dimension
        represent_user_slots,
        represent_agent_slots,
        disease_index,
        last_agent_action,
        last_user_action
    ], dim=0)


    return final_representation



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
    def __init__(self, input_dims, alpha = 0.003 , fc1_dims=128):
        super(Critic, self).__init__()
        
        # Embedding layer for diseases
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

class Actor(nn.Module):
    def __init__(self, n_actions, input_dims, alpha = 0.003, fc1_dims=128):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
#%% Agent Class (networks and memory, take states return actions)
class Agent:
  def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.003, gae_lambda=0.95,
        policy_clip=0.2, batch_size=10, n_epochs=10, symptoms_list = None, disease_list = None, vf_coeff = 0.1879601677, ent_coeff = 0.006057):
    self.gamma = gamma
    self.policy_clip = policy_clip
    self.n_epochs = n_epochs
    self.gae_lambda = gae_lambda
    self.max_turn = 20
    self.n_actions = n_actions
    #complete the difinition of the actor critic parameters
    self.vf_coeff = vf_coeff
    self.ent_coeff = ent_coeff
    self.actor= Actor(n_actions, input_dims, alpha)
    self.critic = Critic(input_dims, alpha)
    self.memory = PPOMemory(batch_size)
    self.symptoms_list = symptoms_list
    self.disease_list = disease_list

  def remember(self, state, action, probs, vals, reward, done):
    self.memory.store_memory(state, action, probs, vals, reward, done)
# choosing action with threshold and confidence
  def choose_action(self, observation, turn):

    dist = self.actor(observation)
    value = self.critic(observation)
    action = dist.sample()
    action= action.item()
    action_tensor = torch.tensor([action], dtype=torch.int64).to(self.actor.device)
    probs = torch.squeeze(dist.log_prob(action_tensor)).item()
    value = torch.squeeze(value).item()
    act_as_dict = self.wrap_action(action_list[action])


    return action, act_as_dict, probs, value
  
  def expert_action(self, observation):
      
      probs = 1.0  # Since this is the expert action, you can assume it has the highest probability
      value = self.critic(observation)
      
      value = torch.squeeze(value).item()
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
  
      if action in symptoms_list:
        return {'action': 'request', 'request_slots': {action: "" }, 'disease': 'UNK'}
      else:
        return {'action': 'inform', 'request_slots': {},'disease': action}
      
    
  def initialize_episode(self, example):
     self.implicit_slots = list(example['implicit_inform_slots'].keys())
     self.current_dis = example['disease_tag']

  def learn(self):
      p = True
      for _ in tqdm(range(self.n_epochs)):
          
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
              
              states = torch.vstack([state_arr[i] for i in batch]).float().to(self.actor.device)
              
              old_probs = torch.tensor([old_prob_arr[i] for i in batch]).to(self.actor.device)
              
              actions = torch.tensor([action_arr[i] for i in batch], dtype=torch.int64).to(self.actor.device)

              dist = self.actor(states)
              critic_value = self.critic(states)

              critic_value = torch.squeeze(critic_value)

              new_probs = dist.log_prob(actions)
              prob_ratio = new_probs.exp() / old_probs.exp()
              
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
              total_loss.backward(retain_graph=True)
              self.actor.optimizer.step()
              self.critic.optimizer.step()

      self.memory.clear_memory()


class Manager:
  def __init__(self, dataset= None, action_list =None ,symptoms_list=None, disease_list=None, input_dim = None, config=None ):
    self.dataset = dataset
    self.action_list = action_list
    self.symptoms_list = symptoms_list
    self.disease_list = disease_list
    self.input_dim = input_dim
    self.action_space_size = len(action_list)
    alpha = config['alpha']
    batch_size = config['batch_size']
    self.N = config['N_steps']
    ppo_clip = config['policy_clip']
    gamma = config['gamma']
    gae_lambda = config['gae_lambda']
    vf_coeff = config['vf_coeff']
    ent_coeff = config['entropy_coefficient']
    self.supervised = False # 1 for expert action, 0 for agent action

    positive_r= config['positive_reward']
    negative_r = config['negative_reward']
    asking_r = config['asking_reward']

    self.user = user(symptoms_list, action_list, disease_list, input_dim, positive_r, negative_r, asking_r)

    self.agent = Agent(n_actions=self.action_space_size, batch_size=batch_size, 
               alpha=alpha, gamma=gamma, gae_lambda=gae_lambda, policy_clip=ppo_clip,
               input_dims=self.input_dim, vf_coeff=vf_coeff, ent_coeff=ent_coeff)
    self.state_tracker = StateTracker(symptoms_list, disease_list)
    self.train_episodes = int(0.8*len(dataset))
    self.test_episodes = len(dataset) - self.train_episodes

    self.hit = 0
    self.n_expert_episodes = 50
  def run_episodes(self):
    self.hit = 0

    best_score = -np.inf
    score_history = []
    
    n_steps = 0
    avg_score = 0

    for i in range(self.train_episodes):
      done = False
      score = 0
      self.state_tracker.reset()

      observation, example = self.user._reset_()
      self.state_tracker.update(user_action=observation)
      observation = self.state_tracker.get_state()
      self.agent.initialize_episode(example)
      if (i+1) % self.n_expert_episodes == 0:
        self.supervised = False

      while not done:
        n_steps += 1
        if self.supervised:
          action, act_as_dict ,prob, val = self.agent.expert_action(observation)

        else:
          action, act_as_dict, prob, val = self.agent.choose_action(observation, n_steps)


        self.state_tracker.update(agent_action=act_as_dict)
        
        observation_, reward, dialogue_status, done = self.user._step_(act_as_dict)



        self.state_tracker.update(user_action=observation_)


        score += reward

        input_state = self.state_tracker.get_state()
        self.agent.remember(input_state, action, prob, val, reward, done)
      
        if n_steps % self.N == 0:
            self.agent.learn()
            
        
        observation = self.state_tracker.get_state()
    
      if dialogue_status:
        self.hit += 1



      score_history.append(score)
      avg_score = np.mean(score_history[-100:])

      if avg_score > best_score:
          best_score = avg_score
 
  def test_accuracy(self):

    self.hit = 0
    for _ in range(self.test_episodes):
      done = False
      score = 0
      self.state_tracker.reset()
      observation, example = self.user._reset_()
      self.state_tracker.update(user_action=observation)
      observation = self.state_tracker.get_state()
      self.agent.initialize_episode(example)
      n_steps = 0
      while not done:
        n_steps += 1
        action, act_as_dict, prob, val = self.agent.choose_action(observation, n_steps)
        self.state_tracker.update(agent_action=act_as_dict)
        observation_, reward, dialogue_status, done = self.user._step_(act_as_dict)
        self.state_tracker.update(user_action=observation_)
        score += reward
        input_state = self.state_tracker.get_state()
        observation = input_state

      if dialogue_status:
        self.hit += 1

    return self.hit / self.test_episodes

#%% Helper Functions
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
  'asking_reward' : -3,
  'alpha': 0.0037,
  'gamma': 0.92,
  'vf_coeff': 0.1879601677,
  'entropy_coefficient': 0.06057,
  'policy_clip': 0.2375,
  'gae_lambda': 0.814,
  'batch_size': 128,
  'N_steps' : 256,
  'n_epochs': 10

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

    input_dims =  3* num_of_symptoms + 3
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
    print("Test Accuracy: ", manager.test_accuracy())
    return accuracy



#%% hypertune using optuna
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
