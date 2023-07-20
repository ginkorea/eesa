from preprocessing.data import *

df = read_jsonl('data/newsmtsc-mt-hf/train.jsonl')
select = df[['polarity', 'sentence']]
print(select)