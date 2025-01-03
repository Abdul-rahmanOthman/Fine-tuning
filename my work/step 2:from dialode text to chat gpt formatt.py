import json

def convert_dialogue_format(input_data_list):
    results = []

    for input_data in input_data_list:
        dialogue = input_data["dialogue"]
        true_disease = input_data["true_disease"]

        messages = []
        for line in dialogue:
            role, content = line.split(": ", 1)
            role = "user" if role == "User" else "assistant"
            messages.append({"role": role, "content": content})

        messages.append({"role": "assistant", "content": f"true_disease: {true_disease}"})
        results.append({"messages": messages})

    return results

input_data_list = [
    {
        "dialogue": [
            "User: I have cough and runny nose.",
            "Model: Have you been sneeze?",
            "User: No",
            "Model: Have you been allergy?",
            "User: Yes"
        ],
        "true_disease": "allergic rhinitis"
    },
    {
        "dialogue": [
            "User: I have a headache.",
            "Model: Have you taken any medication?",
            "User: Yes",
            "Model: Did it help?",
            "User: No"
        ],
        "true_disease": "migraine"
    }
]

output_data_list = convert_dialogue_format(input_data_list)

# Format the output to ensure each message is on a new line and compact
formatted_output_list = []
for data in output_data_list:
    formatted_output = json.dumps(data, separators=(',', ':'))
    formatted_output_list.append(formatted_output)

# Join each dialogue set with a new line
final_output = "\n".join(formatted_output_list)
print(final_output)
