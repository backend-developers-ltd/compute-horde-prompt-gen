import logging

from prompt import PROMPT_ENDING

log = logging.getLogger(__name__)


class MockModel:
    def __init__(self):
        pass

    def generate(self, prompts: list[str], num_return_sequences: int, **_kwargs):
        return [1 for _ in range(len(prompts) * num_return_sequences)]

    def decode(self, _output):
        return f"COPY PASTE INPUT PROMPT {PROMPT_ENDING} Here is the list of prompts:\nHow are you?\nDescribe something\nCount to ten\n"


class GenerativeModel:
    def __init__(self, model_path: str, quantize: bool = False):
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )

        quantization_config = None
        if quantize:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                llm_int8_enable_fp32_cpu_offload=False,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            log.info("using quantized model")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        # set default padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompts: list[str],
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
    ):
        # encode the prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        return self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,  # use sampling-based decoding
        )

    def decode(self, output):
        return self.tokenizer.decode(output, skip_special_tokens=True)
