import data.data
from preprocessing.feature_extraction import *
from preprocessing.language_processing import *
from llm.sentiment import *
from data.data import *
import pandas as pd


class Pipe:

    def __init__(self, dataset):
        self.raw = load(dataset)
        self.processed = None
        self
