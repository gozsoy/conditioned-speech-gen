import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse


import gensim.downloader as api
import pickle
import argparse


word_embedding = {
    'glove': "glove-wiki-gigaword-300",
    'word2vec': "word2vec-google-news-300"
    }

def create_enc_dict(file_name, embedding, task):

    embedding_file = word_embedding[embedding]
    if task == 'key2article':
        folder_name = file_name
    else:
        folder_name = os.path.dirname(file_name)

    print('file_name: ', file_name)
    print('folder_name: ', folder_name)
    print('word_embedding: ', embedding)

    ######## Load word embedding data
    print('{} word embeddings loading...'.format(embedding))
    encoder = api.load(embedding_file)
    print('{} word embeddings loaded'.format(embedding))
    glove_dict = {}

    if not task == 'key2article':
        file1 = open(file_name, "r+")
        lines = file1.readlines()

        i=0
        for line in lines:
            # below is original
            #keywords = list(line.strip().split(", "))

            # below is my addition with in_text as first keyword
            # old approach where separator is ,
            #keywords = list(line.strip().split(", "))[1:]
            
            # separator ||
            intext_keyword_list = list(line.strip().split('|| '))
            keywords = list(intext_keyword_list[1].split(", "))
            

            #print(keywords)
            for word in keywords:
                glove_dict[word] = encoder[word]

            # save_path = folder_name + '/' + str(embedding) + '_set_' +str(i) + '.npy'
            # np.save(save_path, glove_words)
            i=i+1
    else:
        keyword_sets = []
        for filename in os.listdir(folder_name):
            if filename.endswith('txt'):
                file1 = open(folder_name + filename, "r+")
                lines = file1.readlines()
                keywords = list(lines[2].strip().split(", "))
                in_text = lines[1].split()[:30]
                keyword_sets.append((' '.join(in_text), keywords))
                for word in keywords:
                    glove_dict[word] = encoder[word]

    save_path_dict = folder_name + '/dict_' + str(embedding) + '.pkl'
    with open(save_path_dict, 'wb') as f:
        pickle.dump(glove_dict, f, pickle.HIGHEST_PROTOCOL)

    
if __name__ == "__main__":
    ######## Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str)
    parser.add_argument('-word_embedding', type=str, default='glove',
                            choices=list(word_embedding.keys()), help='word_embedding')
    parser.add_argument('-task', type=str, default=None) #'key2article', 'commongen'
    args = parser.parse_args()
    file_name = args.file
    embedding = args.word_embedding
    task = args.task

    create_enc_dict(file_name, embedding, task)


