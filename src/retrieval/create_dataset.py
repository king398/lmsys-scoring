import datasets
import pandas as pd

dataset = datasets.load_dataset("teknium/OpenHermes-2.5")

all_conversations_list = dataset['train']['conversations']

all_prompts = []
all_responses = []

for i in all_conversations_list:
    for j in i:
        if j['from'] == "human":
            all_prompts.append(j['value'])
        elif j['from'] == "gpt":
            all_responses.append(j['value'])

prompt_response_df = pd.DataFrame({'prompt': all_prompts, 'response': all_responses})

