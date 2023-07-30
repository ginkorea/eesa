from util import *
from openai_utils import *


class SentiChat(openai.ChatCompletion):

    def __init__(self, debug=False):
        super().__init__(engine="gpt-3.5-turbo")
        self.debug = debug
        self.engine = "gpt-3.5-turbo"
        self.intro = [{"role": "system",
                       "content": "You are a sentiment analyzer.  You will respond to a user's prompt with a two "
                                  "four part response seperated by '|'.  The first part will be a sentiment score "
                                  "between -1 and 1.  -1 means the user's prompt is very negative, while 1 means the "
                                  "user's prompt is very positive, a score of 0 is neutral. The second part is your "
                                  "confidence in the rating with 0 being no confidence and 1 being extremely confident."
                                  "The forth part will be a one to two sentence explanation of your reasoning for "
                                  "rating a text as positive, negative or neutral. The third part will be your grade of"
                                  "the explanation with 0 being the worst and 1 being the best. The overall response "
                                  "response format is: Sentiment Score [-1 - 1] | Confidence Rating [0 -1] | "
                                  "Explanation Grade [0 - 1] | Explanation [Free Text]"}]
        self.messages = self.intro
        self.prompt = "Analyze the sentiment of the following:"

    def create(self, text):
        if len(self.messages) > 6:
            self.messages = self.intro + self.messages[-1:]
        response = super().create(
            model=self.engine,
            messages=text,
        )
        if self.debug:
            yellow(response)
        self.messages.append(response["choices"][0]["message"])

    def say_last(self):
        cyan(self.messages[-1]["content"])

    def get_user_input(self):
        self.messages.append({"role": "user", "content": self.prompt + input()})

    def chat(self):
        if self.debug:
            cyan(self.messages)
        self.get_user_input()
        self.create(self.messages)
        self.say_last()


def create_senti_chat_bot():
    openai.api_key = get_openai_key()
    cb = SentiChat()
    cb.say_last()
    run = True
    while run:
        cb.chat()
        if cb.messages[-1]["content"] == "exit":
            run = False


create_senti_chat_bot()
