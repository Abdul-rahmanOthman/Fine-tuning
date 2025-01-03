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

with open(r"C:\Users\\7mood\Desktop\RLProject\Data\dxy_data.txt", encoding="utf-8") as file:
  original_dataset = json.load(file)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
# change default dpi 
plt.rcParams['figure.dpi'] = 350
def plot_two_groups(group1, group2, title="---"):
    pca = PCA(n_components=2)
    labels = [0] * len(group1) + [1] * len(group2)

    print(f" group 1 stats (w len {len(group1)})", np.mean(group1, axis=1), np.std(group1, axis=1))
    print(f" group 2 stats (w len {len(group2)})", np.mean(group2, axis=1), np.std(group2, axis=1))
    
    concatenated_data = np.vstack([group1, group2])
    pca_data = pca.fit_transform(concatenated_data)


    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, alpha= 0.7)
    
    # Add index numbers above each scatter point
    for i, (x, y) in enumerate(pca_data):
        plt.text(x, y, str(i), fontsize=12, ha='center', va='center')
    
    plt.title(title)
    plt.show()



def all_combinations(any_list):
    comb_list = []
    for r in range(1, len(any_list) + 1):
        comb_list.extend(combinations(any_list, r))
    return comb_list

with open(r"C:\Users\\7mood\Desktop\RLProject\Data\disease_list.txt", encoding="utf-8") as file:
  disease_list= []
  for line in file:
    line = line.strip()
    disease_list.append(line)

updated_dataset = list()
with open(r"C:\Users\\7mood\Desktop\RLProject\Data\symptoms_list.txt", encoding="utf-8") as file:
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
from util import create_perfect_agent_dataset, one_hot_encoded_to_embeddings


states, actions, action_type = create_perfect_agent_dataset(original_dataset, num_of_symptoms, symptom_mapping)
# the action type is a list of strings that describe the type of the action, either symptom or disease. 
# I want to group the actions that are symptoms and the disease together. Where the disease is the last action in the group

ix_grouped = [ [] ] # each sublist in ix_grouped is a group of states that are related to each other

for ix, action in enumerate(actions):
  if action_type[ix] == 'disease':
    ix_grouped.append([ix])
  else:
    ix_grouped[-1].append(ix)


states_test, actions_test, action_type_test = create_perfect_agent_dataset(test_list, num_of_symptoms, symptom_mapping)
ix_grouped_test = [[]] # each sublist in ix_grouped is a group of states that are related to each other

for ix, action in enumerate(action_type_test):
  if action_type_test[ix] == 'disease':
    ix_grouped_test.append([ix])
  else:
    ix_grouped_test[-1].append(ix)



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

#%% define a custom torch data loader
import torch
from torch.utils.data import Dataset, DataLoader

import random

class TripletDataset(Dataset):
    def __init__(self, data_tensor, groups):
        self.data_tensor = data_tensor
        groups = [group for group in groups if len(group) > 1]  # Remove groups with only one element

        self.groups = groups

    def __len__(self):
        return len(self.groups)  # Number of groups

    def __getitem__(self, index):
        # Select the group for positive samples
        group = self.groups[index]
        if len(group) < 2:
            raise ValueError("Group should have at least 2 elements")
        # Choose two different random indices from the same group for positive samples
        pos_indices = random.sample(group, 2)
        pos1, pos2 = self.data_tensor[pos_indices[0]], self.data_tensor[pos_indices[1]]

        # Choose a different group for the negative sample
        all_indices = list(range(len(self.groups)))
        all_indices.remove(index)  # Remove the current group index
        neg_group_index = random.choice(all_indices)
        neg_group = self.groups[neg_group_index]
        
        # Choose one random index from the different group
        neg_index = random.choice(neg_group)
        neg = self.data_tensor[neg_index]

        return pos1, pos2, neg
    

# Create dataset
dataset = TripletDataset(states, ix_grouped)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator( device= 'cuda'))
# %%
# Example of accessing the data
for data in dataloader:
    pos1, pos2, neg = data
    print("Pos1:", pos1, "Pos2:", pos2, "Neg:", neg)
    break
# %% 

class sequence(nn.Module):
    def __init__(self, unique_states_with_actions, embedding_dim = 20):
        super(sequence, self).__init__()
        self.embeddings = nn.Embedding(unique_states_with_actions, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 7, batch_first=True, num_layers=5, dropout=0.3)
        self.linear = nn.Linear(7, 7)

        

    def forward(self, x):
        x = self.embeddings(x)
        x, (hidden, cell) = self.lstm(x)
        hidden_layer = hidden[-1] # get the last hidden layer, has the shape of (batch_size, hidden_size)

        x = self.linear(hidden_layer)
        return x

    
import torch.optim as optim

#%% CONSIDER PRESERVING THE INFORMATION IN THE STATE AS WELL SOMEHOW. 


class TripletModelTrainer:
    def __init__(self, unique_states_with_actions = vocab_size, embedding_dim=40):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sequence(unique_states_with_actions, embedding_dim).to(self.device)
        self.criterion = nn.TripletMarginWithDistanceLoss(margin= 10)
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, dataloader, num_epochs=10000, states_test=states_test):
        for epoch in range(num_epochs):
            for data in dataloader:
                anchor, positive, negative = data
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                output_anchor = self.model(anchor)
                output_positive = self.model(positive)
                output_negative = self.model(negative)

                # Calculate loss
                loss = self.criterion(output_anchor, output_positive, output_negative)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
                self.evaluate(states_test)

                self.model.train()


            

            

        print("Training complete!")

    def evaluate(self, states_test):
        self.model.eval()
        embedded_test_states = self.model(states_test)
        embedded_test_states = embedded_test_states.detach().cpu().numpy()

        embedded_test_states_grouped = []
        for group in ix_grouped_test:
            if len(group) < 2:
                continue

            embedded_test_states_grouped.append(embedded_test_states[group])

        plot_two_groups(embedded_test_states_grouped[1], embedded_test_states_grouped[2], title = 'group 1 and 2')
        plot_two_groups(embedded_test_states_grouped[1], embedded_test_states_grouped[6], title = 'group 1 and 6')
        plot_two_groups(embedded_test_states_grouped[3], embedded_test_states_grouped[9], title = 'group 3 and 9')
                

        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    
    trainer = TripletModelTrainer(vocab_size, embedding_dim=20)
    trainer.train(dataloader)
    
    model = trainer.model
    embedded_test_states = model(states_test)
    embedded_test_states = embedded_test_states.detach().cpu().numpy()

    embedded_train_states = model(states)
    embedded_train_states = embedded_train_states.detach().cpu().numpy()

    

    embedded_test_states_grouped = []
    for group in ix_grouped_test:
        if len(group) < 2:
            continue

        embedded_test_states_grouped.append(embedded_test_states[group])

    

    plot_two_groups(embedded_test_states_grouped[9], embedded_test_states_grouped[7])


