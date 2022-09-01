import os
import yake
import random
import shutil
import pandas as pd
import gensim.downloader as api

encoder = api.load("glove-wiki-gigaword-300")

number_of_keywords = 3
max_ngram_size = 1
total_shards = 10
prompt_len = 25  # words

custom_kw_extractor = yake.KeywordExtractor(n=max_ngram_size)

valid_df = pd.read_csv('/cluster/scratch/goezsoy/nlp_lss_datasets/processed_df_valid.csv')

# only extract first 5k validation sentences' keywords because of slow generation
#valid_df = valid_df.iloc[:5000] extract all

with open('valid_df_keywords_20k.txt', 'w') as f:

    for _, row in valid_df.iterrows():

        text = row.speech
        keywords_dict = custom_kw_extractor.extract_keywords(text)

        keywords = list(map(lambda temp_dict: temp_dict[0].lower(), keywords_dict))
        filtered_keywords = list(filter(lambda kw: kw in encoder.key_to_index.keys(), keywords))
        selected_keywords = ', '.join(random.sample(filtered_keywords,min(len(filtered_keywords),number_of_keywords)))

        # no prompt
        if text.split()[0] in ['Mr.','Madam']:
            selected_keywords = text.split(',')[0] + '|| ' + selected_keywords + '\n'
        else:
            selected_keywords = '<|endoftext|>' + '|| ' + selected_keywords + '\n'
        
        # prompt
        #selected_keywords = ' '.join(text.split()[:prompt_len]) + '|| ' + selected_keywords + '\n'

        f.write(selected_keywords)

    f.close()


speech_per_shard = len(valid_df)//total_shards

with open('valid_df_keywords_20k.txt', 'r') as fp:
    
    speech_counter = 0
    shard_counter = 0
    shard_path = None
    
    for line in fp:

        if speech_counter == speech_per_shard:
            shard_counter += 1
            speech_counter = 0
        
        if speech_counter == 0:
            shard_path = os.path.join('data', 'shard'+str(shard_counter))
            if os.path.exists(shard_path):
                shutil.rmtree(shard_path)
            os.mkdir(shard_path)
            
        
        with open(os.path.join(shard_path,'keywords.txt'), 'a') as f:
            f.write(line)
            f.close()
        
        speech_counter += 1