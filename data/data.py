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


def merge_files_stb(path1, path2):
    sentiment_df = pd.read_csv(path1, sep='|', index_col=0, names=['index', 'sentiment_score'])
    sentences_df = pd.read_csv(path2, sep='\t', names=['sentence_index', 'sentence'])
    merged_df = sentiment_df.merge(sentences_df, left_index=True, right_on='sentence_index')
    merged_df.drop(columns=['sentence_index'], inplace=True)
    output_file_path = 'stb.csv'
    merged_df.to_csv(output_file_path, index=False, sep="|")
    print("Saving %s" % output_file_path)


def rename_stb():
    df = pd.read_csv('stb.csv', sep='|')
    rename_columns = {
        'sentiment_score': 'sentiment'
    }
    df.rename(columns=rename_columns, inplace=True)
    df.to_csv('stb.csv', index=False, sep='|')


def sts_gold_tweet():
    file = 'sts_gold_tweet.csv'
    df = pd.read_csv(file)
    df.drop(columns=['id'], inplace=True)
    rename_columns = {
        'polarity': 'sentiment',
        'tweet': 'sentence'
    }
    df.rename(columns=rename_columns, inplace=True)
    var = {0: 0, 4: 1}
    df['sentiment'] = df['sentiment'].map(var)
    output = 'gold.csv'
    df.to_csv(output, index=False, sep="|")


def concat_d():
    try:
        datasets = load_source_data()
        df = pd.concat(datasets, ignore_index=True)
        df.to_csv('combined.csv', index=False, sep='|')
        print("Successfully merged and saved to 'combined.csv'")
    except Exception as e:
        print("Error occurred:", e)


def load_source_data():
    try:

        df1 = pd.read_csv('gold.csv', sep='|')
        df2 = pd.read_csv('stb.csv', sep='|')
        df3 = pd.read_csv('movies.csv', sep='|')
        datasets = [df1, df2, df3]
        return datasets
    except Exception as e:
        print("Error occurred while reading CSV files:", e)
        return []


def load(file):
    df = pd.read_csv(file, sep='|')
    return df


def fix_the_movies(file):
    df = pd.read_csv(file, sep='|')
    df['sentence'] = df['sentence'].str.replace('\n', ' ')
    df_grouped = df.groupby('sentiment')['sentence'].apply(' '.join).reset_index()
    df_grouped.to_csv(f'fixed_{file}', index=False, sep='|')


# fix_the_movies('movies.csv')
df = load('gold.csv')
small_df = df.sample(frac=0.01, random_state=1)
small_df.to_csv('gold_small.csv', index=False, sep='|')