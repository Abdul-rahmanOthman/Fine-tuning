#import libraries
import json
import copy
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
"""
Dataset shape:
    - Each point in the dataset is a dictionary with the following keys:
    
    {
    'request_slots': {'disease': 'UNK'},
    'implicit_inform_slots': \{'sneeze': False, 'allergy': True},
    'explicit_inform_slots': {'cough': True, 'runny nose': True},
    'disease_tag': 'allergic rhinitis',
    'consult_id': 'train-0'
    }
"""

with open(r"C:\\Users\\7mood\\Desktop\\RLProject\\MedicalDiagnosis\\dxy_data.txt") as file:
  data = json.load(file)


with open(r"C:\\Users\\7mood\\Desktop\\RLProject\\MedicalDiagnosis\\disease_list.txt") as file:
  disease_list= []
  for line in file:
    line = line.strip()
    disease_list.append(line)



with open(r"C:\\Users\\7mood\\Desktop\\RLProject\\MedicalDiagnosis\\symptoms_list.txt") as file:
  symptoms_list= []
  for line in file:
    line = line.strip()
    symptoms_list.append(line)

#Read MDD dataset pickle
MDdata = pd.read_pickle(r"C:\Users\\7mood\Desktop\RLProject\Data\train.pk")

with open(r"C:\Users\\7mood\Desktop\RLProject\Data\symptom.txt") as file:
  MDsymptoms= []
  for line in file:
    line = line.strip()
    MDsymptoms.append(line)

with open(r"C:\Users\\7mood\Desktop\RLProject\Data\disease.txt") as file:
  MDdiseases= []
  for line in file:
    line = line.strip()
    MDdiseases.append(line)


#Read Sym3K dataset pickle
symdata = pd.read_pickle(r"C:\Users\\7mood\Desktop\Dataset\SYMCat3K.p")
symdata_train = symdata['train']
symdata_test = symdata['test']

'''
{'consult_id': 1059,
   'disease_tag': 'Central retinal artery or vein occlusion',
   'group_id': '7',
   'goal': {'request_slots': {'disease': 'UNK'},
    'explicit_inform_slots': {'Spots or clouds in vision': True},
    'implicit_inform_slots': {'Diminished vision': True,
     'Symptoms of eye': True,
     'Pain in eye': True}}}
'''

#compute number of symptoms and diseases from dataset of user_goals


class dataset(Dataset):
    def __init__(self, data, symptomsList = None, diseaseList=None) -> None:
        """
        this class is responsible for handling the dataset and the symptoms and diseases list

        """
        self.data = data
        self.symptomsList = symptomsList
        self.diseaseList = diseaseList
        pass

    def get_symptoms_len(self):
        """
        this function is responsible for getting the current symptoms of the environment
        """
        return len(self.symptomsList)
    
    def get_disease_len(self):
        """
        this function is responsible for getting the current disease of the environment
        """
        return len(self.diseaseList)
    
    def get_dataset_len(self):
        """
        this function is responsible for getting the length of the dataset
        """
        return len(self.data)
    
    def symptoms_occurance(self):
        """
        this function is responsible for getting how many each symptom occured in all the dataset points(either explicit or implicit)
        """
        symptoms_occurance = {}
        for point in self.data:
            for symptom in point['implicit_inform_slots']:
                if symptom not in symptoms_occurance:
                    symptoms_occurance[symptom] = 1
                else:
                    symptoms_occurance[symptom] += 1
            for symptom in point['explicit_inform_slots']:
                if symptom not in symptoms_occurance:
                    symptoms_occurance[symptom] = 1
                else:
                    symptoms_occurance[symptom] += 1

        return symptoms_occurance
    
    def visualize_symptoms_occurance(self):
        """
        This function is responsible for visualizing the symptoms occurrence in the dataset using matplotlib.
        The strength of colors indicates the frequency of symptoms.
        """
        symptoms_occurance = self.symptoms_occurance()
        symptoms_occurance = dict(sorted(symptoms_occurance.items(), key=lambda item: item[1], reverse=True))
        
        fig = plt.figure(figsize=(20, 10))
        plt.bar(symptoms_occurance.keys(), symptoms_occurance.values())
        plt.xticks(rotation=90)
        
        # Set the y-axis ticks to be more detailed
        max_occurance = max(symptoms_occurance.values())
        y_ticks = range(0, max_occurance + 10, 10)
        plt.yticks(y_ticks)
        
        plt.show()
        
        return symptoms_occurance
    
    def diseases_occurance(self):
        """
        this function is responsible for getting how many each disease occured in all the dataset points
        """
        diseases_occurance = {}
        for point in self.data:
            disease = point['disease_tag']
            if disease not in diseases_occurance:
                diseases_occurance[disease] = 1
            else:
                diseases_occurance[disease] += 1
        return diseases_occurance

    def visualize_diseases_occurance(self):
        """
        This function is responsible for visualizing the diseases occurrence in the dataset using matplotlib.
        The strength of colors indicates the frequency of diseases.
        """
        diseases_occurance = self.diseases_occurance()
        diseases_occurance = dict(sorted(diseases_occurance.items(), key=lambda item: item[1], reverse=True))
        
        fig = plt.figure(figsize=(20, 10))
        plt.bar(diseases_occurance.keys(), diseases_occurance.values())
        plt.xticks(rotation=90)
        
        # Set the y-axis ticks to be more detailed
        max_occurance = max(diseases_occurance.values())
        y_ticks = range(0, max_occurance + 10, 10)
        plt.yticks(y_ticks)
        
        plt.show()
        
        return diseases_occurance
    
    
    def print_sample(self):
        """
        This function is responsible for printing a sample from the dataset
        """
        print(self.data[0])
    
    def disease_symptom_correspondence(self):
        """
        This function is responsible for getting the disease and its corresponding dependent symptoms
        """
        disease_symptom_correspondence = {}
        for point in self.data:
            disease = point['disease_tag']
            if disease not in disease_symptom_correspondence:
                disease_symptom_correspondence[disease] = set()
            for symptom in point['implicit_inform_slots']:
                #add symptom to set
                disease_symptom_correspondence[disease].add(symptom)
            for symptom in point['explicit_inform_slots']:
                disease_symptom_correspondence[disease].add(symptom)
        return disease_symptom_correspondence
    
    def visualize_disease_symptom_correspondence(self):
        """
        This function is responsible for visualizing the disease and its corresponding dependent symptoms 
        using matplotlib.
        """
        disease_symptom_correspondence = self.disease_symptom_correspondence()
        fig = plt.figure(figsize=(20, 10))
        for disease, symptoms in disease_symptom_correspondence.items():
            plt.bar(disease, len(symptoms))
        plt.xticks(rotation=90)
        plt.show()
        
        return disease_symptom_correspondence
    
    def calculate_symptom_occurrences(self):
        """
        Calculate the number of occurrences of each symptom for each disease.
        """
        symptom_occurrences = {}
        total_size = len(self.data)
        
        for data_point in self.data:
            disease = data_point['disease_tag']
            if disease not in symptom_occurrences:
                symptom_occurrences[disease] = {}
            
            # Combine implicit and explicit inform slots
            inform_slots = {**data_point['implicit_inform_slots'], **data_point['explicit_inform_slots']}
            
            for symptom, present in inform_slots.items():
                if present:
                    if symptom not in symptom_occurrences[disease]:
                        symptom_occurrences[disease][symptom] = 0
                    symptom_occurrences[disease][symptom] += 1
        
        # Convert counts to percentages
        for disease in symptom_occurrences:
            for symptom in symptom_occurrences[disease]:
                symptom_occurrences[disease][symptom] = (symptom_occurrences[disease][symptom] / total_size) * 100
        
        return symptom_occurrences
    
    def visualize_symptom_occurrences_heatmap(self):
        """
        Visualize the symptom occurrences as a heatmap.
        """
        symptom_occurrences = self.calculate_symptom_occurrences()
        
        # Convert the nested dictionary into a DataFrame
        df = pd.DataFrame(symptom_occurrences).fillna(0)
        
        plt.figure(figsize=(20, 10))
        sns.heatmap(df, annot=True, fmt="g", cmap="YlGnBu")
        plt.title("Symptom Occurrences by Disease")
        plt.xlabel("Disease")
        plt.ylabel("Symptom")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()
        
        return df
       




DxyDataset = dataset(data, symptoms_list, disease_list)
#print(DxyDataset.visualize_symptoms_occurance())
print("Dxy disease_symptom_correspondence: ", DxyDataset.visualize_disease_symptom_correspondence())

print("Dxy disease_symptom_dependency:" , DxyDataset.visualize_symptom_occurrences_heatmap())


MDdataset = dataset(MDdata, MDsymptoms, MDdiseases)
print(MDdataset.visualize_symptoms_occurance())
print("MDdataset length: ", MDdataset.get_len_dataset())
print(MDdataset.visualize_diseases_occurance())

#Sym3K = dataset(symdata, sym_symptoms,sym_diseases)

