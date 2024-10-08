import random
from seeds import THEMES, ABILITIES, FORMATS


class PromptGeneratingPrompt:
    def random_select(self, arr: list[str], num: int = 5) -> str:
        random.shuffle(arr)
        return ", ".join(arr[:num]) + ", etc"

    def generate_prompt(self, short=True) -> str:
        themes = self.random_select(THEMES, num=3)

        if short:
            return (
                f"Generate a list of 10 questions or instruct tasks related to the themes of {themes}. "
                f"Output each prompt on a new line without any extra commentary or special characters."
            )

        relevance_level = random.randint(5, 20)
        complexity_level = random.randint(5, 20)

        abilities = self.random_select(ABILITIES, num=4)
        formats = self.random_select(FORMATS, num=5)

        return (
            f"Generate a list of 10 complex prompts (questions or instruct tasks) that cover a wide range of skills and knowledge areas related to the themes of {themes}. "
            f"Each of these prompts should: "
            f"\n- have a complexity level of {complexity_level} out of 20 and a relevance level to the theme of {relevance_level} out of 20"
            f"\n- test various cognitive abilities ({abilities}) and require different types of writting formats ({formats})"
            f"\n- challenge the model's ability to understand and respond appropriately"
            f"\n- varyingly explore the {themes} in a manner that is consistent with their assigned complexity and relevance levels to the theme"
            f"\nOutput each prompt on a new line without any extra commentary or special characters."
        )

    def generate_role(self, short=True) -> str:
        if short:
            return "You are a prompt engineer tasked with prompts of varying complexity to test the capabilities of a new language model."
        return "You are a prompt engineer tasked with prompts of varying complexity to test the capabilities of a new language model. For each prompt, consider what aspect of the language model's capabilities it is designed to test and ensure that the set of prompts covers a broad spectrum of potential use cases for the language model. Only output the prompts, one per line without any extra commentary. Do not use any special characters or formatting, numbering or styling in the output."
