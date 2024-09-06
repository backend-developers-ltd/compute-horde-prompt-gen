import logging
import io

log = logging.getLogger(__name__)


def strip_input(output: str, ending: str) -> str:
    # input prompt is repeated in the output, so we need to remove it
    idx = output.find(ending) + len(ending)
    return output[idx:].strip()


class MockModel:
    def __init__(self):
        pass

    def generate(self, prompts: list[str], num_return_sequences: int, **_kwargs):
        content = f"Here is the list of prompts:\nHow are you?\nDescribe something\nCount to ten\n"
        return [content for _ in range(len(prompts) * num_return_sequences)]


class GenerativeModel:
    def __init__(self, model_path: str, quantize: bool = False):
        self.input_prompt_ending = None

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

    def tokenize(self, prompts: list[str], role: str) -> str:
        # set default padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        role_templates = {
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "end": "<|start_header_id|>assistant<|end_header_id|>",
        }

        def tokenize(prompt: str) -> str:
            msgs = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ]
            full_prompt = io.StringIO()
            for msg in msgs:
                full_prompt.write(role_templates[msg["role"]].format(msg["content"]))
            full_prompt.write(role_templates["end"])
            return full_prompt.getvalue()

        inputs = [tokenize(prompt) for prompt in prompts]
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to("cuda")
        return inputs

    def decode(self, output) -> list[str]:
        return [
            strip_input(
                self.tokenizer.decode(x, skip_special_tokens=True),
                self.input_prompt_ending,
            )
            for x in output
        ]

    def generate(
        self,
        prompts: list[str],
        role: str,
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
    ):
        # encode the prompts
        inputs = self.tokenize(prompts, role)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,  # use sampling-based decoding
        )

        return self.decode(output)


class Phi3(GenerativeModel):
    def __init__(self, model_path: str, quantize: bool = False):
        super().__init__(model_path, quantize)
        self.input_prompt_ending = "assistant<|end_header_id|>"


class Llama3(GenerativeModel):
    def __init__(self, model_path: str, quantize: bool = False):
        super().__init__(model_path, quantize)
        self.input_prompt_ending = " }}assistant"
