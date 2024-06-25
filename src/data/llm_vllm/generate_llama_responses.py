from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv", encoding='utf-8')
df = df[df['fold'] != 0].reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

LLAMA_70B_PROMPT_TEMPLATE = '''
You are an expert language model analyst. Your task is to compare two responses to a given prompt and explain in detail why one is better than the other. Provide a comprehensive analysis based on the following criteria:

1. Relevance (Score 1-10):
   - How well does each response address the original prompt?
   - Are all parts of the prompt adequately covered?

2. Coherence (Score 1-10):
   - Is the response well-structured with a clear flow of ideas?
   - Are there logical transitions between sentences and paragraphs?

3. Clarity (Score 1-10):
   - How easy is it to understand the main points of each response?
   - Is the language used appropriate for the target audience?

4. Depth (Score 1-10):
   - Does the response provide sufficient detail and examples?
   - Are complex ideas thoroughly explained?

5. Accuracy (Score 1-10):
   - Is the information provided factually correct?
   - Are there any misleading or outdated statements?

6. Tone (Score 1-10):
   - Is the tone appropriate for the context of the prompt?
   - Does it maintain a consistent voice throughout?

7. Creativity (Score 1-10):
   - Does the response offer unique insights or perspectives?
   - Is there original thinking demonstrated?

8. Conciseness (Score 1-10):
   - Is the response appropriately concise without omitting important details?
   - Is there unnecessary repetition or verbosity?

Original Prompt:
{original_prompt}

Response A:
{response_a}

Response B:
{response_b}

Better Response: {better_response}

Analysis:
1. Provide a detailed explanation of why the chosen response is better, referencing each of the criteria mentioned above.
2. For each criterion, explain why one response scored higher than the other.
3. Highlight any particular strengths of the better response.
4. Identify specific areas where the other response could be improved.
5. If applicable, mention any unique aspects or approaches in either response that are noteworthy.
6. Whenever you mention a critique or suggestion, quote the text from the responses to support your analysis.

Scoring Summary:
- Response A Scores: [Provide scores for each criterion]
- Response B Scores: [Provide scores for each criterion]
- Total Score A: [Sum of A's scores]
- Total Score B: [Sum of B's scores]

Conclusion:
Summarize in 2-3 sentences why the chosen response is superior overall.
'''
label_to_response = {0: 'A', 1: 'B', 2: 'tie'}

prompt = LLAMA_70B_PROMPT_TEMPLATE.format(original_prompt=df['prompt'][0], response_a=df['response_a'][0],
                                          response_b=df['response_b'][0],
                                          better_response=label_to_response[df['label'][0]])
prompt = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024,top_k=-1,)
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", )
outputs = llm.generate(prompt*4, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
