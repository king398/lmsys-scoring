import random
import pandas as pd
import re
from tqdm import tqdm
import json
from uuid import uuid4

df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv')
df = df[df['fold'] != 0]




prompt_template = """Paraphrase the following conversation, considering the provided label throughout the process. Maintain the essence, overall structure, and tone of each response while using significantly different wording and examples.

{conversation}

Label: {label}

Guidelines for paraphrasing:

1. Preserve the core message and main points of each response.
2. Adjust the content subtly to align with the given label, emphasizing strengths or weaknesses as appropriate.
3. Use fresh examples, analogies, or phrasing that convey similar ideas to the original.
4. Maintain any unique elements like humor, formality, or technical detail characteristic of each response.
5. Ensure the paraphrased version reads naturally and captures the original tone and style.
6. If the label indicates a tie, balance the quality and appeal of both responses equally.
7. The responses should be packed in the following <prompt></prompt>, <response_a></response_a>, <response_b></response_b> format. Inside them should be the paraphrase text. Keep in mind each new conversation inside the original conversation should have its own conversation

Your paraphrased version should feel like a natural conversation that accurately represents the original while subtly reflecting the label's assessment."""

label_to_text = {0: 'A is better', 1: 'B is better', 2: 'Both are equal'}


def generate_jsonl_entry(conversation, label):
    formatted_prompt = prompt_template.format(conversation=conversation, label=label)
    return {
        "custom_id": str(uuid4()),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt}
            ],
            "temperature": 1.1,
            "seed": random.randint(0, 1000),
            "max_tokens": 16384
        }
    }


# Write JSONL file
with open('openai_batch_requests.jsonl', 'w') as f:
    for i in tqdm(range(100)):
        convo = df.iloc[i]['text']
        label = label_to_text[df.iloc[i]['label']]
        jsonl_entry = generate_jsonl_entry(convo, label)
        f.write(json.dumps(jsonl_entry) + '\n')

print("JSONL file 'openai_batch_requests.jsonl' has been created.")
