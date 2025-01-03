import openai

# Set your OpenAI API key
openai.api_key = 'your secret_key from fine-tuned model '

# Define your fine-tuned model name
fine_tuned_model = 'ft:gpt-3.5-turbo-0125:personal:try1:9cafuiCb'

# List of dialogues to test
dialogues = [
    "Hi, I have a headache and a fever.",
    "My stomach hurts and I feel nauseous.",
    "I'm having trouble breathing and my chest feels tight."
]

# Function to get model output for each dialogue
def get_model_output(dialogues):
    responses = []
    for dialogue in dialogues:
        response = openai.ChatCompletion.create(
            model=fine_tuned_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": dialogue}
            ],
            max_tokens=100  # Adjust the max tokens as needed
        )
        responses.append(response['choices'][0]['message']['content'].strip())
    return responses

# Get and print the model outputs
outputs = get_model_output(dialogues)
for i, output in enumerate(outputs):
    print(f"Dialogue {i+1}: {dialogues[i]}")
    print(f"Model Output: {output}")
    print("="*50)
