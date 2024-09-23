import datetime
import os
import logging
import argparse

from prompt import PromptGeneratingPrompt
from model import MockModel, Llama3, Phi3
from utils import parse_output, append_to_file

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generate_prompts(
    model,
    total_prompts,
    batch_size: int = 5,
    num_return_sequences: int = 5,
    max_new_tokens: int = 2000,
    temperature: float = 1.0,
    filepath: str = "prompts.txt",
):
    prompt_generator = PromptGeneratingPrompt()

    i = -1
    while total_prompts > 0:
        i += 1
        prompts = [prompt_generator.generate_prompt() for _ in range(batch_size)]
        role = prompt_generator.generate_role()

        start_ts = datetime.datetime.now()
        sequences = model.generate(
            num_return_sequences=num_return_sequences,
            prompts=prompts,
            role=role,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        seconds_taken = (datetime.datetime.now() - start_ts).total_seconds()

        new_prompts = []
        for j, sequence in enumerate(sequences):
            generated_prompts = parse_output(sequence)
            new_prompts.extend(generated_prompts)

        # check_prompts_quality(new_prompts)

        # remove any duplicates
        new_prompts = list(set(new_prompts))
        log.info(
            f"{i=} generation took {seconds_taken:.2f}s; generated {len(new_prompts)} prompts"
        )
        if total_prompts - len(new_prompts) < 0:
            # one might want to optimize here and save additional prompts for next batch,
            # but it is so parametrized that it produces on average additional 10 prompts
            # so to fill 240 we would nedd 24 runs - each 10s to produce additional batch
            # for free - saving 10s - so about 4% gain - not worth it :)
            new_prompts = new_prompts[:total_prompts]

        total_prompts -= len(new_prompts)
        append_to_file(new_prompts, filepath)

        if total_prompts == 0:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=262,  # on A6000 we want 240 prompts generated in single file, but not all results are valid
        help="Batch size - number of prompts given as input per generation request",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,  # better to generate as many as possible prompts on different themes
        help="Number of return sequences outputted for each prompt given as input",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,  # 40 new tokens is enough for reasonable length prompt - 30 caused too much cut off prompts
        help="Max new tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["llama3", "phi3", "mock"],
        required=True,
        help="Model to use - options are llama3 or phi3",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./saved_models/",
        help="Path to load the model and tokenizer from",
    )
    parser.add_argument(
        "--number_of_prompts_per_batch",
        type=int,
        default=240,
        help="Number of prompts per uuid batch",
    )
    parser.add_argument(
        "--uuids",
        type=str,
        required=True,
        help="Comma separated list of uuids, used as file names of output batches, i.e. `output/prompts_{uuid}.txt`",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="/output/",
        help="Folder path to save the generated prompts to",
    )

    args = parser.parse_args()

    uuids = args.uuids.split(",")

    model_path = os.path.join(args.model_path, args.model_name)
    if args.model_name == "mock":
        model = MockModel()
    elif args.model_name == "llama3":
        model = Llama3(
            model_path=model_path,
            quantize=args.quantize,
        )
    elif args.model_name == "phi3":
        model = Phi3(
            model_path=model_path,
            quantize=args.quantize,
        )
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    for uuid in uuids:
        start_ts = datetime.datetime.now()
        generate_prompts(
            model,
            total_prompts=args.number_of_prompts_per_batch,
            batch_size=args.batch_size,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            filepath=os.path.join(args.output_folder_path, f"prompts_{uuid}.txt"),
        )
        seconds_taken = (datetime.datetime.now() - start_ts).total_seconds()
        log.info(
            f"Finished generating {uuid} batch with {args.number_of_prompts_per_batch} prompts in {seconds_taken:.2f} seconds"
        )
