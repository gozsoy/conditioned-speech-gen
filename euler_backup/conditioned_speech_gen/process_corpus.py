import re
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# 1- READ RAW CORPUS
raw_corpus_path = '/cluster/work/lawecon/Work/cmarangon/fnc-politicians-language/data/proc/congress-records/tokenized/'

speech_path_list = glob.glob(raw_corpus_path + 'tokenized-legislator-html-congress-CREC*')

agg_df = pd.DataFrame(columns=['bioguide_id','term_party','speech'])

for temp_path in tqdm(speech_path_list):
    
    df = pd.read_pickle(temp_path)
    df = df.replace({'term_party': {'Democrat': 'D', 'Republican': 'R'}})
    df = df[['bioguide_id','term_party','speech']]
    
    agg_df = pd.concat([agg_df,df], ignore_index=True)

agg_df.to_csv(path_or_buf='/cluster/scratch/goezsoy/nlp_lss_datasets/full-corpus.csv',index=False)
print('raw corpus reading done')


# 2- PREPROCESSING

df = pd.read_csv('/cluster/scratch/goezsoy/nlp_lss_datasets/full-corpus.csv')

df = df[['bioguide_id','term_party','speech']]

print(f'initial speech count: {len(df)}')

# step 1: eleminate duplicate speeches, and remove '/n'
unique_speeches_nested = df.groupby('bioguide_id')['speech'].agg(lambda x: list(set(x.values)))

unique_speeches_single_list = []
for idx,row in unique_speeches_nested.reset_index(drop=False).iterrows():
    for temp_speech in row.speech:
        temp_speech = temp_speech.replace('\n','')
        unique_speeches_single_list.append((row.bioguide_id,temp_speech))

unique_speeches_single_df = pd.DataFrame(unique_speeches_single_list,columns=['bioguide_id','speech'])

print(f'speech count after step 1: {len(unique_speeches_single_df)}')

# step 2: row[0]!='.' speeches are not speech but text snippets describing the law 
processed_df = unique_speeches_single_df[unique_speeches_single_df.speech.apply(lambda row: row[0]=='.')]


print(f'speech count after step 2: {len(processed_df)}')

# remove '. ' sequence which occurs at the start of each speech
processed_df['speech'] = processed_df['speech'].apply(lambda row: row[2:])

# add speaker metadata to processed speeches using pd.merge
df = df.drop('speech',axis=1)
df = df.drop_duplicates(subset='bioguide_id')
processed_df= processed_df.merge(df,on='bioguide_id',how='inner')

# step 3: remove speeches that belong to >95th or <30th percentiles in terms of word count
processed_df['token_count'] = processed_df['speech'].apply(lambda row: len(row.split()))

max_token_count = processed_df['token_count'].quantile(q=0.95)
min_token_count = processed_df['token_count'].quantile(q=0.3)

processed_df = processed_df[(processed_df['token_count'] < max_token_count) & (processed_df['token_count'] > min_token_count)]

print(f'speech count after step 3: {len(processed_df)}')

# step 4: remove speech which include 'as follows:' sequence for filtering out raw law texts
processed_df = processed_df[processed_df['speech'].apply(lambda row: re.search("as follows:", row) is None)][['term_party','speech']]

print(f'speech count after step 4: {len(processed_df)}')


# 3- SPLIT DATA INTO TRAIN-VALIDATION SETS

# shuffle data before splitting
shuffled_processed_df = processed_df.sample(frac= 1.0, random_state=42)

# 5 percent validation data
valid_ratio = 0.05

valid_size = int(len(shuffled_processed_df) * valid_ratio)

processed_df_train = shuffled_processed_df.iloc[:-valid_size]
processed_df_valid = shuffled_processed_df.iloc[-valid_size:]

# save the processed corpus
processed_df.to_csv('/cluster/scratch/goezsoy/nlp_lss_datasets/processed_df.csv')
processed_df_train.to_csv('/cluster/scratch/goezsoy/nlp_lss_datasets/processed_df_train.csv')
processed_df_valid.to_csv('/cluster/scratch/goezsoy/nlp_lss_datasets/processed_df_valid.csv')