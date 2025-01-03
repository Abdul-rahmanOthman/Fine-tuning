# MD or DXY
import pandas as pd
import json
import numpy as np
import itertools
from itertools import combinations
def all_combinations(any_list):
    comb_list = []
    for r in range(1, len(any_list) + 1):
        comb_list.extend(combinations(any_list, r))
    return comb_list

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

num_of_symptoms = len(symptoms_list)
num_of_diseases = len(disease_list)


import copy

symptom_mapping = {symptom: i for i, symptom in enumerate(symptoms_list)}
disease_mapping = {disease: i for i, disease in enumerate(disease_list)}

def create_perfect_agent_dataset(data_list, num_of_symptoms = num_of_symptoms, symptom_mapping = symptom_mapping, disease_mapping = disease_mapping):
  
  states = []
  actions = []
  action_type = []
  # maybe just change the actions to posssible bad actions 
  empty_state = np.zeros(num_of_symptoms)

  for example in data_list:
    initial_state = copy.deepcopy(empty_state)
    for symptom, value in example['explicit_inform_slots'].items():
      initial_state[symptom_mapping[symptom]] = 2 if value else 1

    
    for symptom in example['implicit_inform_slots']:

      states.append(copy.deepcopy(initial_state))
      actions.append(symptom)
      action_type.append('symptom')
    

    
    for combination in all_combinations(example['implicit_inform_slots']):
      temp_combination_state = copy.deepcopy(initial_state)
      for symptom in combination:
        temp_combination_state[symptom_mapping[symptom]] = 2 if example['implicit_inform_slots'][symptom] else 1

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

states, actions, action_type = create_perfect_agent_dataset(dataset)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


encoded_states = np.array(states)
states_tensor = torch.tensor(states, dtype=torch.float32)
import torch.nn.functional as F

class GumbelSoftmax(nn.Module):
    def __init__(self, temperature):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature

    def sample(self, logits):
        return F.gumbel_softmax(logits, tau=self.temperature, hard=True)
# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, temperature):
        super(Generator, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Linear(64, input_dim)  # Ensure final layer output matches input_dim
        )
        
        self.gumbel_softmax = GumbelSoftmax(temperature)

    def forward(self, x):
        logits = self.fc(x)
        return self.gumbel_softmax.sample(logits)

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size()).to(real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty 

def discriminator_loss(real_outputs, fake_outputs, gradient_penalty, lambda_gp):
    return torch.mean(fake_outputs) - torch.mean(real_outputs) + lambda_gp * gradient_penalty

def generator_loss(fake_outputs):
    return -torch.mean(fake_outputs)
import matplotlib.pyplot as plt

def wasserstein_distance(real_outputs, fake_outputs):
    return torch.mean(fake_outputs) - torch.mean(real_outputs)


def train_gan(discriminator, generator, states, num_epochs=100, batch_size=32, lambda_gp=10):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.9))

    scheduler_g = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=30, gamma=0.1)
    d_losses = []
    g_losses = []
    wasserstein_distances = []   
    for epoch in range(num_epochs):
        for i in range(0, len(states) - 1, batch_size):
            batch_states = states[i:i + batch_size]
            batch_next_states = states[i + 1:i + 1 + batch_size]

            if len(batch_states) < batch_size or len(batch_next_states) < batch_size:
                continue
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            real_inputs = torch.cat((batch_states, batch_next_states), dim=1)
            real_outputs = discriminator(real_inputs)
            # create random input for generator of size similar to states
            random_batch_states = np.random.rand(batch_size, len(states[0]))

            random_batch_states = torch.tensor(random_batch_states, dtype=torch.float32)

            fake_next_states = generator(random_batch_states)
            fake_inputs = torch.cat((random_batch_states, fake_next_states), dim=1)
            fake_outputs = discriminator(fake_inputs)
            
            gradient_penalty = compute_gradient_penalty(discriminator, real_inputs, fake_inputs)
            d_loss = discriminator_loss(real_outputs, fake_outputs, gradient_penalty, lambda_gp)
            
            d_loss.backward()
            d_optimizer.step()
            
            wd = wasserstein_distance(real_outputs, fake_outputs)
            # Train Generator more frequently
            for _ in range(2):  # Train generator more frequently
                g_optimizer.zero_grad()
                
                fake_next_states = generator(batch_states)
                fake_inputs = torch.cat((batch_states, fake_next_states), dim=1)
                fake_outputs = discriminator(fake_inputs)
                g_loss = generator_loss(fake_outputs)
                
                g_loss.backward()
                g_optimizer.step()
        
        scheduler_g.step()
        scheduler_d.step()
        
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        wasserstein_distances.append(wd.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, wd: {wd.item()}')

        # Save generated samples at intervals
        if epoch % 10 == 0:
            fake_samples = generator(batch_states).detach().cpu().numpy()
            # Save or visualize fake_samples as needed

    # Plot losses and Wasserstein distances
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Wasserstein Distance During Training")
    plt.plot(wasserstein_distances, label="Wasserstein Distance")
    plt.xlabel("iterations")
    plt.ylabel("Wasserstein Distance")
    plt.legend()

    plt.show()

input_dim = 41  # Example input dimension, replace with actual dimension
temperature = 0.5  # Example temperature, adjust as needed
generator = Generator(input_dim, temperature)

discriminator = Discriminator(input_dim)

train_gan(discriminator, generator, states_tensor)

