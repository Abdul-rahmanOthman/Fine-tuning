#%%
import json
import copy
import numpy as np
import random

with open(r"C:\Users\\7mood\Desktop\RLProject\Data\dxy_data.txt") as file:
  original_dataset = json.load(file)

import torch
torch.set_default_dtype(torch.float64)

# default torch tensor type to float
import torch

with open(r"C:\Users\\7mood\Desktop\RLProject\Data\disease_list.txt") as file:
  disease_list= []
  for line in file:
    line = line.strip()
    disease_list.append(line)

updated_dataset = list()



with open(r"C:\Users\\7mood\Desktop\RLProject\Data\symptoms_list.txt") as file:
  symptoms_list= []
  for line in file:
    line = line.strip()
    symptoms_list.append(line)

updated_symptom_list = symptoms_list 


num_of_symptoms = len(symptoms_list)
num_of_diseases = len(disease_list)

action_list = symptoms_list + disease_list


class dataset():
    def __init__(self, data, symptomsList, diseaseList, train_or_test_or_both = "both") -> None:
        
        """
        This class is responsible for handling the dataset and the symptoms and diseases list
        """
        # an example of the key-value pair describing if its a train or test example is 'consult_id': 'train-0'
        self.data = data
        if train_or_test_or_both == "train":
            self.data = [example for example in self.data if example ['consult_id'].split('-')[0] == 'train']
        elif train_or_test_or_both == "test":
            self.data = [example for example in self.data if example ['consult_id'].split('-')[0] == 'test']
        elif train_or_test_or_both == "both":
            pass

        self.symptomsList = symptomsList
        self.diseaseList = diseaseList
        
        self.reset()


        pass
   
    def get_state(self):
        """
        get the state of the current example
        """
        if self.current_index >= len(self.data):
            self.current_index = 0
            random.shuffle(self.data)
        

        example = self.data[self.current_index]
        self.current_index += 1
        self.current_example = example
        self.implicit_symptoms = example['implicit_inform_slots'].keys()

        
        # initialize a state with all symptoms set to 0
        state = []
        for symptom in self.symptomsList:
            if symptom in self.current_example['explicit_inform_slots']:
                
                if self.current_example['explicit_inform_slots'][symptom] == True:
                    state.append(1)
                elif self.current_example['explicit_inform_slots'][symptom] == False:
                    state.append(2)

            else:
                state.append(0)
            
            


        return np.array(state)
    


    def get_disease(self):
        return self.disease_mapping[self.current_example['disease_tag']]

   
    def evaluate_symptom(self, symptom_idx):
        assert self.symptom_mapping[self.symptomsList[symptom_idx]] == symptom_idx, "Symptom mapping is incorrect"

        if self.symptomsList[symptom_idx] in self.current_example['implicit_inform_slots']:
            
            if self.current_example['implicit_inform_slots'][self.symptomsList[symptom_idx]] == True:

                return 1
            elif self.current_example['implicit_inform_slots'][self.symptomsList[symptom_idx]] == False:
                return 2

        else:
            return 3
        
         
   
    def evaluate_disease(self, disease_idx):
        assert self.disease_mapping[self.diseaseList[disease_idx]] == disease_idx, "Disease mapping is incorrect"
        if self.diseaseList[disease_idx] == self.current_example['disease_tag']:
            return True
        else:
            return False
        

    def reset(self):

        random.shuffle(self.data)
        self.current_index = 0
        self.action_set = self.symptomsList + self.diseaseList
        self.symptom_mapping = {symptom: idx for idx, symptom in enumerate(self.symptomsList)}
        self.disease_mapping = {disease: idx for idx, disease in enumerate(self.diseaseList)}




import torch
class dataset_embedding():
    def __init__(self, data = original_dataset, symptomsList = symptoms_list, diseaseList = disease_list, train_or_test_or_both = "both") -> None:
        
        """
        This class is responsible for handling the dataset and the symptoms and diseases list
        """
        # an example of the key-value pair describing if its a train or test example is 'consult_id': 'train-0'
        self.data = data
        if train_or_test_or_both == "train":
            self.data = [example for example in self.data if example ['consult_id'].split('-')[0] == 'train']
        elif train_or_test_or_both == "test":
            self.data = [example for example in self.data if example ['consult_id'].split('-')[0] == 'test']
        elif train_or_test_or_both == "both":
            pass

        self.symptomsList = symptomsList
        self.diseaseList = diseaseList
        self.reset()


        pass
   
    def _get_state(self):
        """
        get the state of the current example
        """
        if self.current_index >= len(self.data):
            self.current_index = 0
            random.shuffle(self.data)
        

        example = self.data[self.current_index]
        self.current_index += 1
        self.current_example = example

        
        # initialize a state with all symptoms set to 0
        state = []
        for symptom in self.symptomsList:
            if symptom in self.current_example['explicit_inform_slots']:
                
                if self.current_example['explicit_inform_slots'][symptom] == True:
                    state.append(1)
                elif self.current_example['explicit_inform_slots'][symptom] == False:
                    state.append(2)

            else:
                state.append(0)
            
            


        return np.array(state)
    
    def get_state(self, model):
        state = self._get_state()
        # make it a torch tensor
        state = torch.tensor(state, dtype=torch.long)
        embedded_state = model(state)
        return embedded_state.cpu().detach().to().numpy().astype(np.float64)

         

    def get_disease(self):
        return self.disease_mapping[self.current_example['disease_tag']]

   
    def _evaluate_symptom(self, symptom_idx):
        assert self.symptom_mapping[self.symptomsList[symptom_idx]] == symptom_idx, "Symptom mapping is incorrect"

        if self.symptomsList[symptom_idx] in self.current_example['implicit_inform_slots']:
            
            if self.current_example['implicit_inform_slots'][self.symptomsList[symptom_idx]] == True:

                return 1
            elif self.current_example['implicit_inform_slots'][self.symptomsList[symptom_idx]] == False:
                return 2

        else:
            return 3
        
    
    def evaluate_symptom(self, symptom_idx, model):
        current_state = self._get_state()
        current_state[symptom_idx] = self._evaluate_symptom(symptom_idx)
        current_state = torch.tensor(current_state, dtype=torch.long)

        embedded_state = model(current_state)
        return embedded_state.cpu().detach().to().numpy().astype(np.float64) 
    
        
         
   
    def evaluate_disease(self, disease_idx):
        assert self.disease_mapping[self.diseaseList[disease_idx]] == disease_idx, "Disease mapping is incorrect"
        if self.diseaseList[disease_idx] == self.current_example['disease_tag']:
            return True
        else:
            return False
        

    def reset(self):

        random.shuffle(self.data)
        self.current_index = 0
        self.action_set = self.symptomsList + self.diseaseList
        self.symptom_mapping = {symptom: idx for idx, symptom in enumerate(self.symptomsList)}
        self.disease_mapping = {disease: idx for idx, disease in enumerate(self.diseaseList)}

    def embedding_stats(self, model):
        """
        This function is responsible for getting the maximum and minimum values of the embeddings in the data"
        """
        max_val = float('-inf')
        min_val = float('inf')
        for example in self.data:
            state = self._get_state()
            state = torch.tensor(state, dtype=torch.long)
            embedded_state = model(state)
            max_val = max(max_val, torch.max(embedded_state).item())
            min_val = min(min_val, torch.min(embedded_state).item())
        return max_val, min_val
    

 


dataClass = dataset(original_dataset, symptoms_list, disease_list, train_or_test_or_both = "both")
length = len(dataClass.data)

dataClass_train = dataset(original_dataset, symptoms_list, disease_list, train_or_test_or_both = "train")
length_train = len(dataClass_train.data)

dataClass_test = dataset(original_dataset, symptoms_list, disease_list, train_or_test_or_both = "test")
length_test = len(dataClass_test.data)

#%% 
embedding_class  = dataset_embedding(original_dataset, symptoms_list, disease_list, train_or_test_or_both = "train")
embedding_class_test  = dataset_embedding(original_dataset, symptoms_list, disease_list, train_or_test_or_both = "test")

#%%
import gymnasium as gym
from gymnasium import spaces

best_performing_params = {'learning_rate': 0.0037134961264765898, 'n_steps': 256, 'batch_size': 128, 'gamma': 0.9487210238458803, 
                          'gae_lambda': 0.8136627047202266, 'ent_coef': 0.006057819640776694, 'vf_coef': 0.18796016773643454, 
                          'clip_range': 0.23752465643218776, 'positive_reward': 13, 'negative_reward': -5, 'asking_reward': -3}

class ConsultationEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, symptoms, diseases, dataClass = dataClass):
        
        super(ConsultationEnv, self).__init__()
        print("init")
        self.dataClass = dataClass
        self.symptoms = symptoms  # List of all possible symptoms
        
        self.diseases = diseases  # List of all possible diseases
        
        self.num_symptoms = len(symptoms)
        self.num_diseases = len(diseases)
        
        # Observation space: each symptom can be True, False, or Unknown (-1)
        self.observation_space = spaces.MultiDiscrete([4] * self.num_symptoms)
        # Action space: ask for a symptom (0 to num_symptoms-1) or predict a disease (num_symptoms to num_symptoms + num_diseases - 1)
        self.action_space = spaces.Discrete(self.num_symptoms + self.num_diseases)
        
        self.current_step = 0
        self.max_steps = 20  # Max steps before forcing a prediction
    
    def reset(self, **kwargs):
        """
        1. get new example from the dataset
        2. get the state based on the example        
        """
        
        
        # Initialize state with unknown symptoms
        self.state = self.dataClass.get_state()
        self.current_step = 0
        return self.state, {}
    
    def step(self, action):
        if action < self.num_symptoms:
            # Ask for a symptom
            symptom_idx = action

            # Simulate patient response
            self.state[symptom_idx] = self.dataClass.evaluate_symptom(symptom_idx)
            self.correct_prediction = False


            reward = best_performing_params['asking_reward'] 
              # Minor penalty for asking a question
            
            done = False
        else:
            # Predict a disease
            disease_idx = action - self.num_symptoms
            is_correct = self.dataClass.evaluate_disease(disease_idx)
            
            self.correct_prediction = is_correct
            reward = best_performing_params['positive_reward'] if is_correct else best_performing_params['negative_reward']
            
            
            done = True

        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            
            done = True
        

        
        return self.state, reward, done, False, {}
    


    def render(self, mode='human'):
        crrctPridiction = self.correct_prediction
        self.correct_prediction = False
        return crrctPridiction


#%%

class embEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, symptoms, diseases, model, dataClass = embedding_class):
        super(embEnv, self).__init__()
        
        self.symptoms = symptoms  # List of all possible symptoms
        self.model = model
        self.diseases = diseases  # List of all possible diseases
        self.dataClass = dataClass
        self.num_symptoms = len(symptoms)
        self.num_diseases = len(diseases)
        
        # Observation space: each symptom can be True, False, or Unknown (-1)
        self.observation_space = spaces.Box(low = -100, high = 100, shape = (7,), dtype= 'float32')
        # Action space: ask for a symptom (0 to num_symptoms-1) or predict a disease (num_symptoms to num_symptoms + num_diseases - 1)
        self.action_space = spaces.Discrete(self.num_symptoms + self.num_diseases)
        
        self.current_step = 0
        self.max_steps = 20  # Max steps before forcing a prediction
    
    def reset(self, **kwargs):
        """
        1. get new example from the dataset
        2. get the state based on the example        
        """
        
        
        # Initialize state with unknown symptoms
        self.state = self.dataClass.get_state(model = self.model)
        self.current_step = 0
        
        

        return np.array(self.state, dtype='float32'), {}
    
    def step(self, action):
        
        if action < self.num_symptoms:
            # Ask for a symptom
            symptom_idx = action

            # Simulate patient response (randomly for now)
            self.state = self.dataClass.evaluate_symptom(symptom_idx, model = self.model)
            self.correct_prediction = False
            reward =best_performing_params['asking_reward'] 
              # Minor penalty for asking a question
            done = False
        else:
            # Predict a disease
            disease_idx = action - self.num_symptoms
            is_correct = self.dataClass.evaluate_disease(disease_idx)

            self.correct_prediction = is_correct
            reward = best_performing_params['positive_reward'] if is_correct else best_performing_params['negative_reward']
            done = True

        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            
            done = True


        return np.array(self.state, dtype='float32'), float(reward) , done, False, {}
    


    def render(self, mode='human'):
        crrctPridiction = self.correct_prediction
        self.correct_prediction = False
        return crrctPridiction

