import random
import pandas as pd
from openai import OpenAI
import re
from tqdm import tqdm
import time
from uuid import uuid4
import json

client = OpenAI()

# Load the data
df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv')
df = df[df['fold'] != 0][5100:10000]
new_data = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/openai_convo_data.csv')


# Helper functions
def process_prompts_and_responses_with_regex(text):
    prompt_pattern = re.compile(r'<prompt>(.*?)</prompt>', re.DOTALL)
    response_a_pattern = re.compile(r'<response_a>(.*?)</response_a>', re.DOTALL)
    response_b_pattern = re.compile(r'<response_b>(.*?)</response_b>', re.DOTALL)

    prompts = prompt_pattern.findall(text)
    responses_a = response_a_pattern.findall(text)
    responses_b = response_b_pattern.findall(text)

    prompts = [p.strip() for p in prompts]
    responses_a = [r.strip() for r in responses_a]
    responses_b = [r.strip() for r in responses_b]

    return prompts, responses_a, responses_b


prompt = """
Paraphrase the following conversation, considering the provided label throughout the process. Maintain the essence, overall structure, and tone of each response while using significant  different wording and examples.
{conversation}

Label: {label}

Guidelines for paraphrasing:

1. Preserve the core message and main points of each response and the number of responses.
2. Adjust the content subtly to align with the given label, emphasizing strengths or weaknesses as appropriate.
3. Use fresh examples, analogies, or phrasing that convey similar ideas to the original.
4. Maintain any unique elements like humor, formality, or technical detail characteristic of each response.
5. Ensure the paraphrased version reads naturally and captures the original tone and style.
6. If the label indicates a tie, balance the quality and appeal of both responses equally.
7. The responses should be packed in the following <prompt></prompt>, <response_a></response_a>, <response_b></response_b> format.Inside them should be the paraphrase text. Keep in mind each new conversation inside the original conversation should have its own conversation

Your paraphrased version should feel like a natural conversation that accurately represents the original while subtly reflecting the label's assessment."""

label_to_text = {0: 'A is better', 1: 'B is better', 2: 'Both are equal'}


def create_batch_input(start_idx, batch_size):
    batch_input = []
    for i in range(start_idx, min(start_idx + batch_size, len(df))):
        convo = df.iloc[i]['text']
        label = label_to_text[df.iloc[i]['label']]
        indv_prompt = prompt.format(conversation=convo, label=label)

        request = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": indv_prompt}
                ],
                "temperature": 1.1,
                "seed": random.randint(0, 1000)
            }
        }
        batch_input.append(json.dumps(request))

    return "\n".join(batch_input)


def process_batch_output(output_content):
    global new_data
    for line in output_content.split('\n'):
        if line.strip():
            result = json.loads(line)
            response_content = result['response']['body']['choices'][0]['message']['content']
            prompts, responses_a, responses_b = process_prompts_and_responses_with_regex(response_content)

            try:
                assert len(prompts) == len(responses_a) == len(responses_b)
                new_data = pd.concat([new_data, pd.DataFrame({
                    'prompt': [prompts],
                    'response_a': [responses_a],
                    'response_b': [responses_b],
                    'label': df.iloc[int(result['custom_id'].split('-')[1])]['label'],
                    'id': str(uuid4())
                })], ignore_index=True)
            except AssertionError:
                print(f"Error processing response for {result['custom_id']}")

    return new_data


# Main processing loop
total_price = 0
batch_size = 150
iter = tqdm(range(0, len(df), batch_size))
for start_idx in iter:
    # Create batch input
    batch_input = create_batch_input(start_idx, batch_size)

    # Upload batch input file
    with open("batch_input.jsonl", "w") as f:
        f.write(batch_input)

    batch_input_file = client.files.create(
        file=open("batch_input.jsonl", "rb"),
        purpose="batch"
    )

    # Create batch
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Wait for batch to complete
    while True:
        batch_status = client.batches.retrieve(batch.id)
        if batch_status.status == "completed":
            break
        time.sleep(1)  # Check every minute

    # Retrieve and process results
    output_file = client.files.content(batch_status.output_file_id)
    new_data = process_batch_output(output_file.text)

    # Save progress
    new_data.to_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/openai_convo_data.csv', index=False)

    # Calculate and update total price
    batch_price = 0
    for line in output_file.text.split('\n'):
        if line.strip():
            result = json.loads(line)
            usage = result['response']['body']['usage']
            batch_price += (usage['prompt_tokens'] * 0.075 / 1e6) + (usage['completion_tokens'] * 0.3 / 1e6)

    total_price += batch_price
    iter.set_description_str(f"Batch price: ${batch_price:.2f}, Total price: ${total_price:.2f}")

print(f"Processing complete. Total cost: ${total_price:.2f}")
