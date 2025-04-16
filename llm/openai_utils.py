from llm.key import my_key
import openai


def get_openai_key():
    return my_key


def get_openai_engines():
    engine_list = []
    # TODO: The resource 'Engine' has been deprecated
    # engines = openai.Engine.list()
    for engine in engines.data:
        engine_list.append(engine["id"])
    return engine_list