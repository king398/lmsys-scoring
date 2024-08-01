from transformers import AutoTokenizer, AutoModelForCausalLM
from gptqmodel import GPTQModel, QuantizeConfig
import torch
from datasets import load_dataset
import pandas as pd

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-27b-it", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

# Load the dataset
df = pd.read_csv("data/data/train.csv")


def tokenize(
        prompt, response_a, response_b,
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
    return text


df['text'] = tokenize(df['prompt'], df['response_a'], df['response_b'])
text = df['text'][:512]
# Prepare calibration dataset
calibration_dataset = [tokenizer(text) for text in
                       text]
model.save_pretrained("gemma-2-27b-it")
# Configure quantization
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    static_groups=True,
    sym=True,
    true_sequential=True,
    damp_percent=0.1,
    lm_head=False,

)

# Initialize GPTQModel
gptq_model = GPTQModel.from_pretrained("gemma-2-27b-it", quant_config)
# Quantize the model
gptq_model.quantize(calibration_dataset, tokenizer=tokenizer)

# Save the quantized model
output_dir = "gemma-2-27b-it-quantized-4bit"
gptq_model.save_quantized(output_dir)

# Save the tokenizer
tokenizer.save_pretrained(output_dir)

# Save the score state dict separately
torch.save(model.score.state_dict(), f'{output_dir}/score.bin')

# Load the quantized model
loaded_model = GPTQModel.from_quantized(output_dir)

# Example inference
