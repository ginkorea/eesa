# sentiment.py

import time
from openai import OpenAI
from util import *
from openai_llm.openai_utils import get_openai_key

# Initialize OpenAI client
client = OpenAI(api_key=get_openai_key())


class BaseSentimentChat:
    """
    Base wrapper class around OpenAI ChatCompletion for sentiment-related tasks.
    Handles structured prompting and retries.
    """

    def __init__(self, system_prompt: str, model: str = "gpt-4-1106-preview", debug: bool = False):
        self.model = model
        self.debug = debug
        self.messages = [{"role": "system", "content": system_prompt}]
        self.response = None

    def _create_completion(self, messages):
        """Uses the updated OpenAI API v1+ to create a chat completion."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                if self.debug:
                    yellow(response)
                return response.choices[0].message.content
            except Exception as e:
                red(f"[Error] OpenAI API call failed: {e}")
                time.sleep(1 + attempt)
        raise RuntimeError("OpenAI API failed after multiple retries.")

    def _add_user_message(self, content: str):
        """Adds user input to the message history."""
        self.messages.append({"role": "user", "content": content})

    def get_last_message(self):
        """Returns the content of the last user message."""
        return self.messages[-1]["content"]

    def show_last_message(self):
        """Prints the last message in the chat history (for debugging)."""
        cyan(self.get_last_message())


class SentiChat(BaseSentimentChat):
    """
    A specialized sentiment analysis agent that returns:
    [Sentiment Score | Confidence | Explanation Grade | Explanation]
    """

    def __init__(self, debug: bool = False):
        system_prompt = (
            "You are a sentiment analyzer. You will respond to a user's prompt with a four-part response separated by '|'. "
            "The first part is a sentiment score between -1 and 1. "
            "The second part is your confidence in the score (0 to 1). "
            "The third part is a quality grade of your explanation (0 to 1). "
            "The fourth part is a one- to two-sentence explanation of your reasoning.\n\n"
            "Here are some examples:\n\n"
            "Input: I absolutely love this! It's amazing.\n"
            "Output: 0.95 | 0.98 | 0.9 | The sentence is clearly very positive with strong emotional tone.\n\n"
            "Input: This is the worst service I’ve ever had.\n"
            "Output: -0.9 | 0.95 | 0.92 | The sentence expresses strong dissatisfaction and anger.\n\n"
            "Input: It’s okay, I guess. Not great, not terrible.\n"
            "Output: 0.0 | 0.85 | 0.8 | The sentence is neutral with a hint of indifference.\n\n"
            "Now analyze the following input."
        )
        super().__init__(system_prompt, debug=debug)

    def classify_sentiment(self, text: str, verbose: bool = False) -> str:
        """Classifies sentiment for a given input text."""
        self._add_user_message(f"Input: {text}\nOutput:")
        result = self._create_completion(self.messages)
        if verbose:
            cyan(result)
        return result


class SentiSummary(BaseSentimentChat):
    """
    A summarization agent that condenses multiple sentiment explanations into a concise summary.
    """

    def __init__(self, debug: bool = False):
        system_prompt = (
            "You are a component of a sentiment analyzer. Your job is to summarize a list of explanations "
            "into a single concise 1-2 sentence summary that captures the average sentiment reasoning."
        )
        super().__init__(system_prompt, debug=debug)

    def summarize_explanations(self, explanations: list[str], verbose: bool = False) -> str:
        """
        Summarizes multiple explanation strings into one.
        :param verbose: bool to make output verbose or not.
        :param explanations: List of explanation strings.
        """
        explanations_text = "\n".join(f"- {exp}" for exp in explanations)
        self._add_user_message(f"Provide a 1-2 sentence summary of the following explanations:\n{explanations_text}")
        result = self._create_completion(self.messages)
        if verbose:
            cyan(result)
        return result


def interactive_sentiment_loop():
    """Starts an interactive sentiment classification loop in the terminal."""
    bot = SentiChat(debug=True)
    bot.show_last_message()
    while True:
        user_input = input("Enter text to analyze (or 'exit' to quit):\n> ")
        if user_input.lower().strip() == "exit":
            break
        result = bot.classify_sentiment(user_input, verbose=True)
        print("➤ Result:", result)
