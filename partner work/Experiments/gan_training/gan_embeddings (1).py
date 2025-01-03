#%%
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# set default device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

with open(r"F:\rl project 7amood\RLProject\RLProject\Data\dxy_data.txt", encoding="utf-8") as file:
  original_dataset = json.load(file)

from itertools import combinations


def all_combinations(any_list):
    comb_list = []
    for r in range(1, len(any_list) + 1):
        comb_list.extend(combinations(any_list, r))
    return comb_list

with open(r"F:\rl project 7amood\RLProject\RLProject\Data\disease_list.txt", encoding="utf-8") as file:
  disease_list= []
  for line in file:
    line = line.strip()
    disease_list.append(line)

updated_dataset = list()
with open(r"F:\rl project 7amood\RLProject\RLProject\Data\symptoms_list.txt", encoding="utf-8") as file:
  symptoms_list= []
  for line in file:
    line = line.strip()
    symptoms_list.append(line)

updated_symptom_list = symptoms_list 


num_of_symptoms = len(symptoms_list)
num_of_diseases = len(disease_list)

action_list = symptoms_list + disease_list
#%%  create a dataset for an xgboost classifier to classify whether an action taken by the agent is correct or not

# 1. extract the training data from the original dataset
train_list = [ example for example in original_dataset if example['consult_id'].split('-')[0] == 'train']
test_list = [ example for example in original_dataset if example['consult_id'].split('-')[0] == 'test']
# 2. map the symptoms and diseases to index in the feature vector
symptom_mapping = {symptom: i for i, symptom in enumerate(symptoms_list)}
disease_mapping = {disease: i + num_of_symptoms for i, disease in enumerate(disease_list)}

vocab = {**symptom_mapping, **disease_mapping}
vocab["<PAD>"] = len(vocab)

vocab_size = len(vocab)
#%%

def one_hot_encode(actions, action_types, num_of_symptoms = num_of_symptoms, num_of_diseases = num_of_diseases, symptom_mapping = symptom_mapping, disease_mapping = disease_mapping):
    """
    Takes a list of actions which can be either symptoms or diseases and returns a one-hot encoded vector.
    
    Parameters:
        actions (list): List of action strings (symptoms or diseases).
        action_types (list): Corresponding list indicating whether each action is a 'symptom' or 'disease'.
        num_of_symptoms (int): Total number of unique symptoms.
        num_of_diseases (int): Total number of unique diseases.
        symptom_mapping (dict): Dictionary mapping symptoms to indices.
        disease_mapping (dict): Dictionary mapping diseases to indices, offset by num_of_symptoms.

    Returns:
        np.ndarray: One-hot encoded matrix where each row represents an action.
    """

    one_hot_vector = np.zeros((len(actions), (num_of_symptoms + num_of_diseases)))


    for idx, action in enumerate(actions):
        if action_types[idx] == 'symptom':
            one_hot_vector[idx, symptom_mapping[action]] = 1
        elif action_types[idx] == 'disease':
            one_hot_vector[idx, disease_mapping[action]] = 1
        else:
            raise ValueError('Action type should be either "symptom" or "disease"')
    
    return one_hot_vector


def print_state(state):
  for i, val in enumerate(state):
    if val == 1:
      if i < num_of_symptoms:
        print(updated_symptom_list[i])
      else:
        print(disease_list[i - num_of_symptoms])

#%%
# i want to create a dataset that includes all possible paths that an expert (perfect agent) would take.
def create_perfect_agent_dataset(data_list, num_of_symptoms = num_of_symptoms, symptom_mapping = symptom_mapping, disease_mapping = disease_mapping):
  
  states = []
  actions = []
  action_type = []
  # maybe just change the actions to posssible bad actions 
  empty_state = np.zeros(num_of_symptoms)

  for example in data_list:
    
    initial_state = copy.deepcopy(empty_state)
    for symptom in example['explicit_inform_slots']:
      initial_state[symptom_mapping[symptom]] = 1

    
    for symptom in example['implicit_inform_slots']:

      states.append(copy.deepcopy(initial_state))
      actions.append(symptom)
      action_type.append('symptom')
    

    
    for combination in all_combinations(example['implicit_inform_slots']):
      temp_combination_state = copy.deepcopy(initial_state)
      for symptom in combination:
        temp_combination_state[symptom_mapping[symptom]] = 1

      for symptom in example['implicit_inform_slots']:
        if symptom not in combination:
          states.append(copy.deepcopy(temp_combination_state))
          actions.append(symptom)
          action_type.append('symptom')


      if len(combination) == len(example['implicit_inform_slots']):
        states.append(copy.deepcopy(temp_combination_state))
        actions.append(example['disease_tag'])
        action_type.append('disease')
  return states, actions, action_type




#%%
"""
# create a dataset that includes all possible paths that a complete imperfect agent would take
bad_states = []
bad_actions = []

empty_state = np.zeros((num_of_symptoms + num_of_diseases))

for example in train_list[:1]:
  
  initial_state = copy.deepcopy(empty_state)
  for symptom in example['explicit_inform_slots']:
    initial_state[symptom_mapping[symptom]] = 1

  
  for symptom in symptoms_list:

    if symptom not in example['implicit_inform_slots']:
      bad_states.append(copy.deepcopy(initial_state))
      print_state(initial_state)
      bad_actions.append(symptom)
      print('action:', symptom)

      for disease in disease_list:
        bad_states.append(copy.deepcopy(initial_state))
        print_state(initial_state)
        bad_actions.append(disease)
        print('action:', disease)
  
  
  # create a sublist of all symptoms that are not in the implicit_inform_slots
  bad_symptoms = [symptom for symptom in symptoms_list if symptom not in example['implicit_inform_slots']]

  
  for combination in all_combinations(bad_symptoms):
    temp_combination_state = copy.deepcopy(initial_state)
    for symptom in combination:
      temp_combination_state[symptom_mapping[symptom]] = 1

    for symptom in bad_symptoms:
      if symptom not in combination:
        bad_states.append(copy.deepcopy(temp_combination_state))
        print_state(temp_combination_state)
        bad_actions.append(symptom)
        print('action:', symptom)
        for disease in disease_list:
          bad_states.append(copy.deepcopy(temp_combination_state))
          print_state(temp_combination_state)
          bad_actions.append(disease)
          print('action:', disease)

    if len(combination) == len(bad_symptoms):


      for disease in disease_list:
        if disease != example['disease_tag']:
          bad_states.append(copy.deepcopy(temp_combination_state))
          print_state(temp_combination_state)
          bad_actions.append(disease)
          print('action:', disease)

"""

# real world images are like the perfect action given the state 
# the generater should try to generate the perfect action given the state
# the discriminator should try to classify whether the action is perfect or not
# the generator should try to fool the discriminator
# the generator should try to generate the perfect action given the state
# the discriminator should try to classify whether the action is perfect or not

# TRAIN ON TWO PHASES

#phase 1:
  # 1. TRAIN THE GENERATOR TO GENERATE THE PERFECT ACTION GIVEN THE STATE
  # 2. TRAIN THE DISCRIMINATOR TO CLASSIFY WHETHER THE ACTION IS PERFECT OR NOT
  # 3. TRAIN THE GENERATOR TO FOOL THE DISCRIMINATOR

#phase 2:
  # 1. change the state slightly to make the state more ambiguous
  # 2. repeat phase 1

### borrow concept from adversarial learning where we assume the state is:
  # 1. a state generated by an adversarial enviroment

class EmbeddingLayer(nn.Module):
    def __init__(self, unique_states_with_actions, embedding_dim = 7):
        super(EmbeddingLayer, self).__init__()
        self.embeddings = nn.Embedding(unique_states_with_actions, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()    
    # RNN to process sequences of embeddings
    self.rnn = nn.RNN(input_size=7, hidden_size=32, num_layers=1, batch_first=True)
    self.regularization = nn.Dropout(0.2)
    
    # Linear layers follow the RNN output
    self.fc1 = nn.Linear(32, 1)  # Adjust input dimension to match RNN's hidden_size
    

  def forward(self, state_emb):
    
    
    # Passing the embedded sequence to RNN
    # Output will be (output, h_n) where output is all hidden states, and h_n is the last hidden state
    rnn_out, _ = self.rnn(state_emb)
    
    # Using the last hidden state to make the final decision
    # rnn_out[:, -1, :] gets the last RNN output for each batch item
    x = rnn_out[:, -1, :]  # [batch_size, 16]
    x = self.regularization(x)
    
    x = F.sigmoid(self.fc1(x))
    


    return x
  
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.rnn = nn.RNN(input_size=7, hidden_size=64, num_layers=4, batch_first=True)

    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 16)
    self.regualarization = nn.Dropout(0.2)
    self.fc4 = nn.Linear(16, 7)

  def forward(self, state_emb):

    
    #state_emb_padded = pad_sequence(state_emb, batch_first=True, padding_value= vocab["<PAD>"])
    rnn_out, _ = self.rnn(state_emb)
    x = rnn_out[:, -1, :]
    x = self.regualarization(x)
    x = F.leaky_relu(self.fc2(x))

    x = torch.tanh(self.fc3(x))
    x = self.fc4(x)

    return x
  

class GAN:
  def __init__(self, num_of_unique_actions):
    self.generator = Generator()
    self.discriminator = Discriminator()
    self.criterion = nn.BCELoss()
    self.criterion_acts = nn.HingeEmbeddingLoss()
    self.optimizer_generator = torch.optim.SparseAdam(self.generator.parameters())
    self.optimizer_discriminator = torch.optim.SGD(self.discriminator.parameters(), lr= 1e-5)

    self.embedding = EmbeddingLayer(num_of_unique_actions)
    self.optimizer_embedding = torch.optim.SparseAdam(self.embedding.parameters(), lr= 1e-5)

    self.discriminator.to('cuda')
    self.generator.to('cuda')
    self.embedding.to('cuda')


  def _train(self, loader):
    self.discriminator.train()
    self.generator.train()
    self.embedding.train()

    for i, (states, actions) in enumerate(loader):  
      states = states.to('cuda')  
      actions = actions.to('cuda')

      # Zero gradients for the discriminator
      self.optimizer_discriminator.zero_grad()

      # Real data processing

      real = torch.hstack((states, actions))

      real_embed = self.embedding(real) # has the shape of (batch_size, 11, 7)

      real_pred = self.discriminator(real_embed)
      real_loss = self.criterion(real_pred, torch.ones((real_pred.size(0), 1), device='cuda'))
      embedding_loss1A = self.criterion(real_pred, torch.ones((real_pred.size(0), 1), device='cuda'))

      # Generate fake data
      #fake_actions = confidences_states_to_embeddings(fake_actions, action_or_state = "action")

      states_embeddings = self.embedding(states) # has the shape of (batch_size, 10, 7)
      fake_actions_embedded = self.generator(states_embeddings) # has the shape of (batch_size, 7)

      # unsqueeze the actions to have the shape of (batch_size, 1, 7)
      fake_actions_embedded = fake_actions_embedded.unsqueeze(1) # has the shape of (batch_size, 1, 7)
      fake = torch.cat((states_embeddings, fake_actions_embedded), dim=1) # has the shape of (batch_size, 11, 7)

      fake_pred = self.discriminator(fake)

      fake_loss = self.criterion(fake_pred, torch.zeros((fake_pred.size(0), 1), device='cuda'))
      embedding_loss1B = self.criterion(fake_pred, torch.zeros((fake_pred.size(0), 1), device='cuda'))
      d_loss = (real_loss + fake_loss)
      
      if i % 1 != 0: 
        d_loss.backward()
        self.optimizer_discriminator.step()


      # Zero gradients for the generator
      self.optimizer_generator.zero_grad()
      #fake_actions = confidences_states_to_embeddings(fake_actions, action_or_state = "action")

      states_embeddings = self.embedding(states) # has the shape of (batch_size, 10, 7)
      fake_actions_embedded = self.generator(states_embeddings) # has the shape of (batch_size, 7)

      fake_actions_embedded = fake_actions_embedded.unsqueeze(1) # has the shape of (batch_size, 1, 7)
      fake = torch.cat((states_embeddings, fake_actions_embedded), dim=1) # has the shape of (batch_size, 11, 7)

      # calculate a loss between the fake actions and the real actions
      #generator_loss = self.criterion(fake, actions)
      
      fake_pred = self.discriminator(fake)
      g_loss = self.criterion(fake_pred, torch.ones((fake_pred.size(0), 1), device='cuda'))
      embedding_loss2 = self.criterion(fake_pred, torch.ones((fake_pred.size(0), 1), device='cuda'))
      actions_embedded = self.embedding(actions)
      g_loss_fromOptimal = self.criterion_acts(fake_actions_embedded, actions_embedded)
      g_loss_total = g_loss + g_loss_fromOptimal
      g_loss_total.backward()
      self.optimizer_generator.step()

      if i % 5 == 0:
        # Zero gradients for the embedding
        self.optimizer_embedding.zero_grad()
        # calculate the embedding loss
        embedding_loss = embedding_loss1A + embedding_loss1B + embedding_loss2
        embedding_loss.backward()
        self.optimizer_embedding.step()

    return d_loss.item() , g_loss.item()

  def train(self, loader, loader_test, epochs=10):
    for epoch in range(epochs):
      d_loss, g_loss = self._train(loader)
      print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
      print(f"Accuracy on test set: {self.evaluate(loader_test)}")
      print(f"Accuracy on fooled discriminator: {self.evaluate_fooled_discriminator(loader_test)}")
      print(f"Accuracy on random state action pairs: {self.evaluate_on_random_state_action_pairs(loader_test)}")

      print(f"Accuracy on train set: {self.evaluate(loader)}")
      print(f"Accuracy on fooled discriminator: {self.evaluate_fooled_discriminator(loader)}")
      print(f"Accuracy on random state action pairs: {self.evaluate_on_random_state_action_pairs(loader)}")


          
  def generate_actions(self, states):
    self.generator.eval()
    self.embedding.eval()
    with torch.no_grad():
        states = states.to('cuda')
        states_embeddings = self.embedding(states)
        action = self.generator(states_embeddings)
        return action
  

  def evaluate(self, loader):
    self.discriminator.eval()

    self.embedding.eval()

    total = 0
    correct = 0
    for states, actions in loader:
      states = states.to('cuda')
      actions = actions.to('cuda')

      total_state = torch.hstack((states, actions))
      total_state = self.embedding(total_state)


      pred = self.discriminator(total_state)
      pred = pred > 0.5
      correct += torch.sum(pred).item()
      total += len(pred)

    return correct / total

  
  def evaluate_fooled_discriminator(self, loader):
    self.generator.eval()
    self.discriminator.eval()
    self.embedding.eval()
    
    total = 0
    correct = 0

    for states, actions in loader:
      states = states.to('cuda')
      actions = actions.to('cuda')

      fake_actions_embedded = self.generate_actions(states)
      states_embeddings = self.embedding(states)
      fake_actions_embedded = fake_actions_embedded.unsqueeze(1)
      fake = torch.cat((states_embeddings, fake_actions_embedded), dim=1)

      pred = self.discriminator(fake)
      pred = pred > 0.5
      correct += torch.sum(pred).item()
      total += len(pred)

    return 1 - (correct / total)
  
  def evaluate_on_random_state_action_pairs(self, loader):
    self.discriminator.eval()
    self.embedding.eval()
    
    # Assuming your device is configured as 'cuda' as per your previous usage
    device = 'cuda'
    for states, actions in loader:
      states = states.to(device)

      # Generating random actions; adjust this based on your specific action space
      # For discrete actions, you might want to shuffle or randomly sample action labels
      if actions.dtype == torch.int64:  # Assuming actions are categorical
          random_actions = torch.randint_like(actions, high=actions.max()+1)  # Randomly sampled within the same range
      else:  # Assuming actions are continuous
          random_actions = torch.rand_like(actions)  # Randomly sampled from a uniform distribution

      random_actions = random_actions.to(device)

      # Embedding both states and randomly generated actions
      total_state = torch.hstack((states, random_actions))
      embedded_state_action = self.embedding(total_state)

      # Get predictions from the discriminator
      pred = self.discriminator(embedded_state_action)
      pred = pred > 0.5  # Thresholding to decide on fake (0) or real (1)

      # Counting the number correctly identified as fake
      correct = (pred == 0).sum().item()  # Counting how many were correctly identified as fake
      total = pred.size(0)

      return correct / total

    


#%% generate datasets for train and test
"""
states, actions, action_type = create_perfect_agent_dataset(train_list)
states_test, actions_test, action_type_test = create_perfect_agent_dataset(test_list)

actions_train = one_hot_encode(actions, action_type)
actions_test = one_hot_encode(actions_test, action_type_test)

# conver to torch tensors
states = torch.tensor(states)
actions = torch.tensor(actions_train)

# When creating the DataLoader, if you use a generator for randomness:
generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')

# Create DataLoaders
loader = torch.utils.data.DataLoader(list(zip(states, actions_train)), batch_size=32, shuffle=True, generator=generator)
loader_test = torch.utils.data.DataLoader(list(zip(states_test, actions_test)), batch_size=32, shuffle=False, generator = generator)  # No need to shuffle test data
"""

#%% embed the actions and states
def one_hot_encoded_to_embeddings(states):


  if type(states[0]) == str:
    raise ValueError("states should be a list of one hot encoded vectors")

  states_embeddings = []
  for state in states:
    temp = []
    for ix, val in enumerate(state):
      if val:
        temp.append(ix)
    states_embeddings.append(temp)
  return states_embeddings


"""
states could be one of three things:
  1. a list of one hot encoded vectors -> use one_hot_encoded_to_embeddings
  2. a list of strings -> use vocab to map to indices
  3. a list of indices -> use directly
  4. a list of confidences -> use a threshold to determine whether to include the action or not

"""
def confidences_states_to_embeddings(states, threshold = 0.5, states_lenghts = 10, action_or_state = "state"):
  if action_or_state == "state":
    all_states = []
    for state in states:
      temp = []
      for ix, val in enumerate(state):
        if val >= threshold:
          temp.append(ix)

      all_states.append(temp)
    

    # pad the first state to the states_lenghts
    all_states[0] = all_states[0].extend([vocab["<PAD>"]] * (states_lenghts - len(all_states[0])))

    all_states = [torch.tensor(state, dtype=torch.long) for state in all_states]
    all_states = [s.to("cuda") for s in all_states]
    all_states = pad_sequence(all_states, batch_first=True, padding_value=vocab["<PAD>"])

  elif action_or_state == "action":

    all_states = []
    for state in states:
      # take the highest confidence action
      temp = torch.argmax(state)
      
      all_states.append(temp)


    
    all_states = [torch.tensor(state, dtype=torch.long) for state in all_states]
    all_states = [s.to("cuda") for s in all_states]
    all_states = [state.unsqueeze(0) for state in all_states]


  #turn the list of tensors to a tensor
  all_states = torch.stack(all_states)
    


  return all_states



states, actions, action_type = create_perfect_agent_dataset(train_list)
states_test, actions_test, action_type_test = create_perfect_agent_dataset(test_list)



states_embed = one_hot_encoded_to_embeddings(states)
states_test_embed = one_hot_encoded_to_embeddings(states_test)

# convert states_embed to a numpy array
states_np = [np.array(state) for state in states_embed]
max_len = max([len(state) for state in states_np])
states_np = [np.pad(state, (0, max_len - len(state)), 'constant', constant_values=(vocab["<PAD>"])) for state in states_np]



actions_embed = [vocab[action] for action in actions]
actions_test_embed = [vocab[action] for action in actions_test]

#%% #ConverT to torch tensors
states = [torch.tensor(i, dtype=torch.long) for i in states_embed]
states = [s.to("cuda") for s in states]
states = pad_sequence(states, batch_first=True, padding_value=vocab["<PAD>"])

actions = [torch.tensor(action, dtype=torch.long) for action in actions_embed]
actions = [a.to("cuda") for a in actions]


#take the lenght of the train states
states_train_lenght = len(states[0])

# add a dummy example from the train states to the test states
states_test_embed.append(states[0])

states_test = [torch.tensor(state, dtype=torch.long) for state in states_test_embed]
states_test = [s.to("cuda") for s in states_test]

states_test = pad_sequence(states_test, batch_first=True, padding_value=vocab["<PAD>"])

# remove the dummy example from the test states
states_test = states_test[:-1]

actions_test = [torch.tensor(action, dtype=torch.long) for action in actions_test_embed]
actions_test = [a.to("cuda") for a in actions_test]

num_of_unique_states = len(vocab)
num_of_unique_actions = len(disease_mapping)
#%% create a loader for the embeddings
#

actions = [a.unsqueeze(0) for a in actions]
actions_test = [a.unsqueeze(0) for a in actions_test]

generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')

loader = torch.utils.data.DataLoader(list(zip(states, actions)), batch_size=8, shuffle=True, generator=generator)
loader_test = torch.utils.data.DataLoader(list(zip(states_test, actions_test)), batch_size=8, shuffle=False, generator = generator)  # No need to shuffle test data
#%%
train_generator = GAN(num_of_unique_states)
train_generator.train(loader, loader_test, epochs=10000)
