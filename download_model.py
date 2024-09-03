import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

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
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./src/compute_horde_prompt_gen/saved_models/",
        help="Path to save the model and tokenizer to",
    )

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # either give token directly or assume logged in with huggingface-cli
        token=args.huggingface_token or True,
    )
    model.save_pretrained(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.model_path)
