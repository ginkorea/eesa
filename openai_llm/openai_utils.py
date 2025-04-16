# openai_utils.py
import openai
from openai_llm.key import my_key


def get_openai_key():
    """Returns the stored OpenAI API key."""
    return my_key


def initialize_openai_api():
    """Initializes the OpenAI API with the provided key."""
    openai.api_key = get_openai_key()


def get_openai_models():
    """
    Returns a list of available OpenAI models.
    Note: Not all accounts have access to all models.
    """
    models = openai.models.list()
    return [model.id for model in models.data]
