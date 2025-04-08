# src/inference/llm_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLM:
    def __init__(self, model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=32768,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
local_llm = LocalLLM(model_name="LGAI-EXAONE/EXAONE-Deep-2.4B", device=device)