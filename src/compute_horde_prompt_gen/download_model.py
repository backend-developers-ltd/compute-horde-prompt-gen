import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

MODEL_PATHS = {
    "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "phi3": "microsoft/Phi-3.5-mini-instruct",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save huggingface model")
    parser.add_argument(
        "--huggingface_token",
        type=str,
        required=True,
        help="Huggingface token to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["llama3", "phi3"],
        required=True,
        help="Model to use - options are llama3 or phi3",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./saved_models/",
        help="Path to save the model and tokenizer to",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model",
        default=False,
    )

    args = parser.parse_args()
    save_path = os.path.join(args.save_path, args.model_name)
    model_name = MODEL_PATHS[args.model_name]
    print(f"Saving {model_name} model to {save_path}")

    quantization_config = None
    if args.quantize:
        import torch
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("using quantized model")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # either give token directly or assume logged in with huggingface-cli
        token=args.huggingface_token or True,
        quantization_config=quantization_config,
    )
    model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=args.huggingface_token or True,
    )
    tokenizer.save_pretrained(save_path)
