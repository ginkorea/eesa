import pandas as pd
import json


def read_jsonl(file):
    with open(file) as f:
        lines = f.read().splitlines()
    df_temp = pd.DataFrame(lines)
    df_temp.columns = ['json_element']
    df_temp['json_element'].apply(json.loads)
    df = pd.json_normalize(df_temp['json_element'].apply(json.loads))
    return df


def get_columns(df, columns):
    selected = df[[columns]]
    return selected
