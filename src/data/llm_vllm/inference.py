import vllm
import torch

llm = vllm.LLM(
    "gemma-2-9b",
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=1024,
)
tokenizer = llm.get_tokenizer()

text = ["Hello, my dog is cute"]
print(torch.tensor(llm.encode(text,vllm.SamplingParams())[0].outputs.embedding).softmax(-1))