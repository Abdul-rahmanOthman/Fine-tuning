import jsonlines
import json
from google.colab import files
import io

def convert_to_dialogue(data):
    dialogue = []
    true_disease = data.get("disease_tag", "Unknown")

    # User stating explicitly informed symptoms
    informed_symptoms = [f"{symptom.replace('_', ' ')}" for symptom, value in data.get("explicit_inform_slots", {}).items() if value]
    if informed_symptoms:
        user_symptoms = " and ".join(informed_symptoms)
        dialogue.append(f"User: I have {user_symptoms}.")

    # Model asking about all relevant symptoms, including those marked as absent
    for symptom, value in data.get("implicit_inform_slots", {}).items():
        model_ask = f"Have you been {symptom.replace('_', ' ')}?"
        user_response = "Yes" if value else "No"
        dialogue.append(f"Model: {model_ask}")
        dialogue.append(f"User: {user_response}")

    # Additional specific requests (if any)
    for symptom, value in data.get("request_slots", {}).items():
        if value == "UNK" and symptom != "disease" and symptom not in data.get("implicit_inform_slots", {}):
            model_ask = f"Have you been {symptom.replace('_', ' ')}?"
            user_response = "Yes" if data.get("implicit_inform_slots", {}).get(symptom, False) else "No"
            dialogue.append(f"Model: {model_ask}")
            dialogue.append(f"User: {user_response}")

    return {
        "dialogue": dialogue,
        "true_disease": true_disease
    }

# Upload file
uploaded = files.upload()

# Load and process data from uploaded JSON file
file_name = next(iter(uploaded))  # This gets the first uploaded file key
data = json.load(io.BytesIO(uploaded[file_name]))

# Handling both single dictionary and list of dictionaries
results = []
if isinstance(data, list):  # If it's a list, process each item
    results = [convert_to_dialogue(item) for item in data]
else:  # Otherwise, process the single dictionary
    results = [convert_to_dialogue(data)]

# Save results to a .jsonl file
with jsonlines.open('results.jsonl', mode='w') as writer:
    writer.write_all(results)  # Use write_all to write multiple items

# Download the .jsonl file
files.download('results.jsonl')
