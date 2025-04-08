# src/inference/llm_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class LocalLLM:
    def __init__(self, model_name="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate(self, prompt, streaming=False):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
        if streaming:
            streamer = TextIteratorStreamer(self.tokenizer)
            thread = Thread(target=self.model.generate, kwargs=dict(
                input_ids=input_ids.to(self.device),
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                streamer=streamer
            ))
            thread.start()

            for text in streamer:
                print(text, end="", flush=True)
            return text
        
        else:
            output = self.model.generate(
                input_ids.to(self.device),
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )
            return self.tokenizer.decode(output[0])

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
local_llm = LocalLLM(model_name="LGAI-EXAONE/EXAONE-Deep-2.4B", device=device)