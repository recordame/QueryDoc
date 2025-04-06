# src/inference/llm_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLM:
    def __init__(self, model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                top_k=50
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
local_llm = LocalLLM(model_name="LGAI-EXAONE/EXAONE-Deep-2.4B", device=device)