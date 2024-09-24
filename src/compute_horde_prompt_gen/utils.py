import re
import os
import logging
import collections

log = logging.getLogger(__name__)


def clean_line(line: str) -> str:
    line = line.strip()
    head, sep, tail = line.partition("<|")
    if head:
        line = head.strip()
    else:
        # if we started with a tag we assume that inside we find our prompt
        line = tail.partition("|>")[2].partition("<|")[0].strip()
    # remove list numbering if present
    line = re.sub(r"^\s*\d+\.?\s*", "", line)
    # strip quotations
    line = line.strip("\"'")
    return line


def parse_output(output: str) -> list[str]:
    # split into lines and clean them
    lines = output.split("\n")
    for line in lines:
        cleaned_line = clean_line(line)
        # we skip if line is too short or too long and not ends with ?
        # in most cases it would be just first line
        if (
            len(cleaned_line) > 10
            and len(cleaned_line) < 300
            and cleaned_line.endswith("?")
        ):
            return [cleaned_line]

    return []


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
