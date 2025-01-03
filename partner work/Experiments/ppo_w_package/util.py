# i want to create a dataset that includes all possible paths that an expert (perfect agent) would take.
import copy
import numpy as np
def create_perfect_agent_dataset(data_list, num_of_symptoms, symptom_mapping):
  
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

from itertools import combinations

def all_combinations(any_list):
    comb_list = []
    for r in range(1, len(any_list) + 1):
        comb_list.extend(combinations(any_list, r))
    return comb_list

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
