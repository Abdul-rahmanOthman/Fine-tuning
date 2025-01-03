#%% initial setup
#%matplotlib inline

import json
import copy
import random
import numpy as np
import torch

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
"""
1. data as idx -> to go to the embedding layer
2. data as one hot encoded -> to calculate the loss

"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np
import time
# default dpi to 300
plt.rcParams['figure.dpi'] = 300

class LiveLossPlot:
    def __init__(self):
        self.losses = []

    def add_loss(self, loss, title = None):
        self.losses.append(loss)
        clear_output(wait=True)  # Clear the previous plot
        plt.figure(figsize=(8, 5))  # Define a figure size if needed
        
        plt.plot(self.losses, label='Loss', marker='o', markersize= 3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if title is not None:
            plt.title(f'{title} - Loss: {loss:.4f}')
        else:
            plt.title(f'live loss plot - current Loss: {loss:.4f}')
        plt.legend()
        plt.grid(True)
    # Annotating the difference between the last two losses
        if len(self.losses) > 1:
            # Calculate the difference
            loss_diff = self.losses[-1] - self.losses[-2]
            # Prepare the annotation text
            annotation_text = f'Difference: {loss_diff:.4f}'
            # Place annotation on the plot
            plt.annotate(annotation_text, xy=(len(self.losses) - 1, self.losses[-1]),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')

        plt.show()


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# set default device to cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

with open(r"F:\rl project 7amood\RLProject\RLProject\Data\dxy_data.txt", encoding="utf-8") as file:
  original_dataset = json.load(file)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
# change default dpi 
plt.rcParams['figure.dpi'] = 350




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

 
# 1. the whole possible variations of the symptoms values (i.e., initial, explicit_available, explicit_unavailable, \
#   implicit_available, implicit_unavailable, implicit_unknown) 


symptoms_list_with_variations = []
for symptom in symptoms_list:
    symptoms_list_with_variations.append(symptom + "+default") # 0
    symptoms_list_with_variations.append(symptom + "+explicit_available") # 1
    symptoms_list_with_variations.append(symptom + "+explicit_unavailable") # 2
    symptoms_list_with_variations.append(symptom + "+implicit_available") # 3
    symptoms_list_with_variations.append(symptom + "+implicit_unavailable") # 4
    symptoms_list_with_variations.append(symptom + "+implicit_unknown") # 5
    number_of_variations = 6


vocabulary = { word: idx for idx, word in enumerate(symptoms_list_with_variations) }
inverse_vocabulary = { idx: word for word, idx in vocabulary.items() }
vocabulary_size = len(vocabulary)
status_codes = { "default": 0, "explicit_available": 1, "explicit_unavailable": 2, "implicit_available": 3, "implicit_unavailable": 4, "implicit_unknown": 5}


"""
example of the dataset
{'request_slots': {'disease': 'UNK'},
 'implicit_inform_slots': {'sneeze': False, 'allergy': True},
 'explicit_inform_slots': {'cough': True, 'runny nose': True},
 'disease_tag': 'allergic rhinitis',
 'consult_id': 'train-0'}
"""


def define_initial_state(example):
    symptoms_current_status = []
    state = []
    for symptom in symptoms_list:
        if symptom in example['explicit_inform_slots']:
            if example['explicit_inform_slots'][symptom]:
                symptoms_current_status.append("explicit_available")
            else:
                symptoms_current_status.append("explicit_unavailable")


        else:
            symptoms_current_status.append("default")

    
    for symptoms, symptom_current_status in zip(symptoms_list, symptoms_current_status):
        current_variation = symptoms + "+" + symptom_current_status
        state.append(vocabulary[current_variation])
    
    return state


def change_state_based_on_feedback(state, symptomToChange, feedback):
    """
    state: list of idxs from the vocabulary
    symptomToChange: the symptom to change its status
    feedback: its either (implicit_available, implicit_unavailable, implicit_unknown)

    """

    new_state = copy.deepcopy(state)
    symptom_idx = symptoms_list.index(symptomToChange)

    # only change it if it was default
    symptom_status = inverse_vocabulary[state[symptom_idx]].split("+")[1]
    if symptom_status == "default":
        new_state[symptom_idx] = vocabulary[symptomToChange + "+" + feedback]

    return new_state



def get_OHE_representation(state):
    """
    state: list of idxs from the vocabulary
    return: (examples, num_of_symptoms, array of one hot encoded vectors of the possible variations of the symptoms)
    """
    one_hot_encoded = np.zeros((num_of_symptoms, number_of_variations))
    for i, idx in enumerate(state):
        current_symptom = inverse_vocabulary[idx]
        symptom, status = current_symptom.split("+")
        code = status_codes[status]
        one_hot_encoded[i, code] = 1

    return one_hot_encoded


def get_symptoms_variations_code(state):
    """
    state: list of idxs from the vocabulary
    return: (num_of_symptoms,)
    """
    codes = []
    for i, idx in enumerate(state):
        current_symptom = inverse_vocabulary[idx]
        symptom, status = current_symptom.split("+")
        codes.append(status_codes[status])

    return codes







### decoder output possibilities
# 1. one hot encoded
# 2. the embedding of each idx in the state -> the loss will be calculated based on the embedding






#%% define a custom torch data loader
import torch
from torch.utils.data import Dataset, DataLoader
import random

class autoencoderDataset(Dataset):

    def __init__(self, original_data):
        self.data = original_data
        self.epochs_passed = 0
        self.random_factors_
        




    def __len__(self):
        return len(self.data) 

    def __getitem__(self, index):
        # increase the number of epochs passed when the index is 0
        if index == 0:
            self.epochs_passed += 1
        


        initial_state = define_initial_state(self.data[index])
        if self.epochs_passed % 1 == 0:
            self.random_factors_

        for _ in range(self.randomness_factor):
            initial_state = change_state_based_on_feedback(initial_state, self.random_symptom, self.random_feedback)


        #return torch.tensor(initial_state).to(device), torch.tensor(get_OHE_representation(initial_state)).to(device)
        return torch.tensor(get_OHE_representation(initial_state)).to(device), torch.tensor(get_OHE_representation(initial_state)).to(device)

    
    @property
    def random_factors_(self):
        self.randomness_factor = random.choice(list(range(0, 40)))
        self.random_symptom = random.choice(symptoms_list)
        self.random_feedback = random.choice(["implicit_available", "implicit_unavailable", "implicit_unknown"])

        return self.randomness_factor, self.random_symptom, self.random_feedback


train_dataset = [ example for example in original_dataset if example['consult_id'].startswith("train")]

dataset_autoencoder = autoencoderDataset(train_dataset)
batch_size= 32
dataloader_autoencoder= DataLoader(dataset_autoencoder, batch_size= batch_size, shuffle=True, generator=torch.Generator( device= 'cuda'))


#%% encoder arch
sequence_length = 41

class Encoder(nn.Module):
    def __init__(self, encoder_kwargs):
        input_dim, embedding_dim = encoder_kwargs['vocab_size'], encoder_kwargs['embedding_dim']
        super(Encoder, self).__init__()
        #input shape: (batch_size, seq_len)
        #self.embedding = nn.Embedding(input_dim, embedding_dim, device = device, sparse = False) # output shape: (batch_size, seq_len, embedding_dim)
        embedding_dim = number_of_variations

        self.fc = nn.ModuleList([nn.Linear(embedding_dim, 20) for _ in range(sequence_length)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(16) for _ in range(sequence_length)])
        self.dropout2 = nn.ModuleList([nn.Dropout(0.1) for _ in range(sequence_length)])

        self.fc2 = nn.Linear(20 * sequence_length, 15 * sequence_length, device = device)
        self.bn3 = nn.BatchNorm1d(10 * sequence_length)
        self.dropout3 = nn.Dropout(0.1)

        self.fc2_5 = nn.Linear(15 * sequence_length, 10 * sequence_length, device = device)


        self.fc3 = nn.Linear(10 * sequence_length, 20, device = device)

    def forward(self, src):
        #embedded = self.embedding(src)
        embedded = src
        # embedded shape: (batch_size, seq_len, embedding_dim)
        outputs = []
        for i in range(sequence_length):
            
            out = self.fc[i](embedded[:, i, :]) # was tanh # Apply each fully connected layer
            #out = self.bn2[i](out)  # was uncommented

            outputs.append(out)
        
        output = torch.cat(outputs, dim=1)  # Concatenate along the new sequence dimension

        # embedded shape: (batch_size, seq_len * embedding_dim)
        hidden = self.fc2(output) # was sigmoid

        hidden = self.fc2_5(hidden) # was tanh

        hidden_final = self.fc3(hidden)
        # hidden shape: (batch_size, hidden_dim)

        return hidden_final


encoder_kwargs = {
    "input_dim": len(symptoms_list),
    "embedding_dim": 20,
    "hidden_dim": 7,
    "num_layers": 2,
    "dropout": 0.3,
    'vocab_size': vocabulary_size
}

# test the encoder
encoder = Encoder(encoder_kwargs).to(device)

# get first batch from the dataloader
first_batch = next(iter(dataloader_autoencoder))

hidden_final = encoder(first_batch[1])


#%% decoder arch
class Decoder(nn.Module):
    def __init__(self, encoder_kwargs, hidden_dim = 20):

        super(Decoder, self).__init__()
        num_repeats = encoder_kwargs["input_dim"]
        input_dim = encoder_kwargs["hidden_dim"]
        output_dim = 20
        self.num_repeats = num_repeats
        num_classes = number_of_variations
        self.fc = nn.ModuleList([nn.Linear(output_dim, 20 * num_classes) for _ in range(sequence_length)])
        #batchnormalization 
        self.bn_1 = nn.ModuleList([nn.BatchNorm1d(20 * num_classes) for _ in range(sequence_length)])
        #dropout

        self.fcIntermediate = nn.ModuleList([nn.Linear(20 * num_classes, 15 * num_classes) for _ in range(sequence_length)])
        self.bn_2 = nn.ModuleList([nn.BatchNorm1d(15 * num_classes) for _ in range(sequence_length)])

        self.fcIntermediate2 = nn.ModuleList([nn.Linear(15 * num_classes, 10 * num_classes) for _ in range(sequence_length)])

        self.fc2 = nn.ModuleList([nn.Linear(10 * num_classes, num_classes) for _ in range(sequence_length)])
 
    def forward(self, x):
        outputs = []
        for i in range(sequence_length):
            out = self.fc[i](x) # was tanh # Apply each fully connected layer
            out = self.bn_1[i](out)
            out = self.fcIntermediate[i](out) # was sigmoid
            out = self.bn_2[i](out)
            out = self.fcIntermediate2[i](out) # was tanh
            out = torch.softmax(self.fc2[i](out), dim=1)
            outputs.append(out.unsqueeze(1))
        x = torch.cat(outputs, dim=1)  # Concatenate along the new sequence dimension
        
        
        
        return x

# test the decoder
decoder = Decoder(encoder_kwargs, 20).to(device)
decoded = decoder(hidden_final)
decoder_kwargs = {
    "decoder_output_dim": encoder_kwargs["hidden_dim"],
    "output_dim": encoder_kwargs["embedding_dim"],
    "hidden_dim": 20,
    "num_layers": 2,
    "dropout": 0.2,

}


optimizer_kwargs = {
    "lr": 1e-3
}

#%% run training
def assertions(decoder_reconstruction):
    """
    decoder_reconstruction: (batch_size, seq_len, num_classes)
    
    """
    batch_size = decoder_reconstruction.shape[0]
    assert decoder_reconstruction.shape[1] == 41
    assert decoder_reconstruction.shape[2] == 6

    #assert that the num_classes probabilities sum to 1
    assert torch.allclose(torch.sum(decoder_reconstruction, dim=2), torch.ones(batch_size, 41)), "The probabilities of the classes should sum to 1"
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
plotter = LiveLossPlot()
class autoencoderTrainer():
    def __init__(self, encoder_kwargs, decoder_kwargs, optimizer_kwargs):
        self.encoder = Encoder(encoder_kwargs).to(device)
        self.decoder = Decoder(encoder_kwargs).to(device)

        total_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())

        self.optimizer = torch.optim.Adam(total_parameters, **optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=4)

        self.criterion = nn.BCELoss(reduction= 'mean')
        
    def train(self, iterator, num_epochs = 300):
        self.encoder.train()
        self.decoder.train()
        losses = []
        for _ in tqdm(range(num_epochs)):
            epoch_loss = 0
            for i, (src, label) in enumerate(iterator):
                self.optimizer.zero_grad()
                hidden = self.encoder(src)
                output = self.decoder(hidden)
                if i == 0:
                    assertions(output)

                # flatten the output and the target
                output = output.view(-1, output.shape[-1])
                label = label.view(-1, label.shape[-1])

                random_predictions = torch.rand(output.shape[0], 6, requires_grad=False)

                loss = self.criterion(output, label)
                random_loss = self.criterion(random_predictions, label)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            self.scheduler.step(epoch_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            losses.append(epoch_loss)
            plotter.add_loss(epoch_loss, title = "random loss: " + str(round(random_loss.item(), 2),) + " current lr: 10e" + str(int(np.log10(current_lr))))


            

        
        return epoch_loss / len(iterator)
    

    def evaluate(self, iterator):
        self.encoder.eval()
        self.decoder.eval()
        epoch_loss = 0

        with torch.no_grad():
            for i, (src, label) in enumerate(iterator):
                hidden = self.encoder(src)
                output = self.decoder(hidden)

                output = output.view(-1, output.shape[-1])
                label = label.view(-1, label.shape[-1])

                loss = self.criterion(output, hidden)
                epoch_loss += loss.item()
        
        return epoch_loss / len(iterator)
    

    def save(self):
        save_model(self)

    def load(self):
        self = load_model(self)
    

def save_model(trainer):
    
    encoder, decoder = trainer.encoder, trainer.decoder
    path = "F:\\rl project 7amood\\RLProject\\RLProject\\models\\"
    torch.save(encoder.state_dict(), path + "encodervlinear.pth")
    torch.save(decoder.state_dict(), path + "decodervlinear.pth")

def load_model(trainer):
    path = "F:\\rl project 7amood\\RLProject\\RLProject\\models\\"
    trainer.encoder.load_state_dict(torch.load(path + "encodervlinear.pth"))
    trainer.decoder.load_state_dict(torch.load(path + "decodervlinear.pth"))



if __name__ == "__main__":
    trainer = autoencoderTrainer(encoder_kwargs, decoder_kwargs, optimizer_kwargs)
    trainer.train(dataloader_autoencoder, num_epochs = 10000)
    trainer.evaluate(dataloader_autoencoder)
    trainer.save()




