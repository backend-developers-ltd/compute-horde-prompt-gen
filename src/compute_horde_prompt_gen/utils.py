import re
import os
import logging
import collections

from prompt import PROMPT_ENDING

log = logging.getLogger(__name__)


def clean_line(line: str) -> str:
    line = line.strip()
    # remove list numbering if present
    line = re.sub(r"^\s*\d+\.?\s*", "", line)
    return line


def parse_output(output: str) -> list[str]:
    # input prompt is repeated in the output, so we need to remove it
    idx = output.find(PROMPT_ENDING) + len(PROMPT_ENDING)
    output = output[idx:].strip()

    # split into lines and clean them
    lines = output.split("\n")
    lines = [clean_line(line) for line in lines]

    # filter out null lines or prompts that are too short or long
    lines = [line for line in lines if (len(line) > 10 and len(line) < 300)]

    # skip first line as that's frequently broken (i.e. "Here are the prompts:")
    return lines[1:]


def check_prompts_quality(prompts: list[str]):
    counter = collections.Counter(prompts)
    freqs = []
    for _, frequency in counter.items():
        if frequency > 1:
            freqs.append(frequency)
    # count the frequency of the frequencies
    freq_counter = collections.Counter(freqs)
    for freq, num in freq_counter.items():
        log.warn(f"Found {num} prompts with {freq} duplicates")
    if not freq_counter:
        log.info("All prompts generated are unique")


def append_to_file(prompts: list[str], filepath: str = "prompts.txt"):
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        with open(filepath, "a") as f:
            for prompt in prompts:
                f.write(prompt + "\n")
        log.info(f"Saved {len(prompts)} prompts to: {filepath}")
    except IOError as e:
        log.error(f"Error while saving prompts: {e}")
