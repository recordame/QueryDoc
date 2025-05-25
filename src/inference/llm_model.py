# src/inference/llm_model.py

from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class LocalLLM:
    def __init__(self, model_name="trillionlabs/Trillion-7B-preview", attn_implementation="flash_attention_2", device="gpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="data/hub",
            padding_side='left',
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            cache_dir="data/hub",
            trust_remote_code=True,
        ).to(self.device)

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
                max_new_tokens=4096,
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
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )
            return self.tokenizer.decode(output[0])


if torch.cuda.is_available():
    device = "cuda"
    attn_implementation = "flash_attention_2"
elif torch.backends.mps.is_available():
    device = "mps"
    attn_implementation = "sdpa"
else:
    device = "cpu"
    attn_implementation = "sdpa"

local_llm = LocalLLM(model_name="trillionlabs/Trillion-7B-preview",
                     attn_implementation=attn_implementation,
                     device=device)
