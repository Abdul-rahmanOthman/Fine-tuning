import json

# Function to convert dialogue format
def convert_dialogue(dialogue):
    converted = []
    for line in dialogue:
        if "User:" in line:
            converted.append(line)
        elif "Model:" in line:
            question = line.split(": ")[1]
            question = question.replace("Have you been", "Have you experienced")
            converted.append(f"Model: {question}")
        else:
            converted.append(line)
    return converted

# Split and save dialogues and true diseases
with open("training text.txt", "r") as infile, open("dialogues.txt", "w") as dialogues_file, open("true_diseases.txt", "w") as diseases_file:
    for line in infile:
        item = json.loads(line.strip())
        dialogue = item["dialogue"]
        true_disease = item["true_disease"]

        converted_dialogue = convert_dialogue(dialogue)
        for dialogue_line in converted_dialogue:
            dialogues_file.write(dialogue_line + "\n")
        dialogues_file.write("\n")  # Add a newline to separate dialogues
        diseases_file.write("true_disease: " + true_disease + "\n")

print("Dialogues and true diseases have been saved to 'dialogues.txt' and 'true_diseases.txt'.")
