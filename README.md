# Semantically Conditioned Language Models for Political Text Generation

We develop a 3-step fully automatized and scalable pipeline for generating high quality synthetic text corpus with semantically conditioned language models. Applying the model on a political speech corpus leaves a performant classifier in high confusion for discriminating real and synthetic texts, proving the ability of our pipeline.

## Reproducing results on ETH Euler cluster

## Install Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Close the current terminal and open a new one.

## Setup Conda Environment, Load Modules, Activate Conda Environment

```
conda env create -f environment.yml
module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
conda activate cond_text_gen_project
```
## Note
You do not need to run each of these steps. However, please make sure that you do the corresponding path changes outlined at each step.
+ If you have access to 'processed_df_train.csv' and 'processed_df_valid.csv' files, skip Step 1.
+ If you have access to pretrained checkpoint such as '18aug_k2t_gpt2medium_maxseqlen256_batch8_8_lr2e5_epoch2.pt', skip Step 2.
+ If all 'shard0, ..., shard9' subfolders in 'data/' folder has 'keyword.txt' file, skip Step 3. (currently like this.)
+ If all 'shard0, ..., shard9' subfolders in 'results/' folder has generated outputs such as 'results/shard0/finetunedgptmed_lr2e5_epoch2/Result_w_5.0_nBeams_1_nGenSent_128_nWordsPerSent_1_topP_0.9_WC_Guar_True_glove_maxSENTENCES.txt', skip Step 4. (currently like this.)
+ If you have access to 'quality_fake_df.csv', skip Step 5.

'resushardx/experiment_name/your_experiment

## Step 1: Preprocess Corpus

We assume that you have access to raw corpus directory indicated in line 8.

Go to process_corpus.py, and change line 9 with your save_dir (e.g. '/cluster/scratch/{eth_username}/nlp_lss_datasets'). If you have access to 'processed_df_train.csv' and 'processed_df_valid.csv' files, put them under this directory.

Then
```
python process_corpus.py
```

## Step 2: Fine-tune GPT-2

Go to config.yml.

Change data_path (e.g. '/cluster/scratch/{eth_username}/nlp_lss_datasets').

Change checkpoint_dir (e.g. '/cluster/scratch/{eth_username}/nlp_lss_checkpoints'). If you have access to finetuned checkpoint, put it under this directory.

Change experiment name and other hyperparameters.

Then run on GPU with
```
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python run.py --config config.yml
```

## Step 3: Extract Keywords

Go to extract_keywords.py.

Change line 8 with your gensim data dir (e.g. os.environ['GENSIM_DATA_DIR']='/cluster/scratch/{eth_username}/nlp_lss_datasets').

Change line 9 with the dir of 'processed_df_valid.csv'. (e.g.  '/cluster/scratch/{eth_username}/nlp_lss_datasets/processed_df_valid.csv')

Change line 10 with your preferred keyword_file_name (e.g. 'valid_df_keywords_20k.txt').

Note total_shard in line 17. Make sure that 'results/' folder has subfolders named shard0, shard1, ..., shard{total_shard-1}. (Default codebase is as such.)

Then
```
python extract_keywords.py
```

## Step 4: Generate Fake Speech with K2T

Go to perplexity.py, and change line 5 with your cache_dir (e.g. '/cluster/scratch/{eth_username}/huggingface_cache'). Cache and gensim data directories needs to be changed because of low memory in $HOME directory.

Go to utility_gpt.py. 

Change line 9 with your gensim data dir (e.g. os.environ['GENSIM_DATA_DIR']='/cluster/scratch/{eth_username}/nlp_lss_datasets'). (It should be the same with step 3.)

Change line 10 with your cache_dir (e.g. '/cluster/scratch/{eth_username}/huggingface_cache'). Cache and gensim data directories needs to be changed because of low memory in $HOME directory.

Change line 11 with your converter_table_path (e.g. '/cluster/scratch/{eth_username}/nlp_lss_datasets/converter_table_glove'). Converter table holds glove word vectors for each token in gpt-2 space.

Go to k2t.py, and change line 744 model checkpoint dir (e.g. '/cluster/scratch/{eth_username}/nlp_lss_checkpoints/18aug_k2t_gpt2medium_maxseqlen256_batch8_8_lr2e5_epoch2.pt').

Then run on GPU with
```
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard0/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard1/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard2/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard3/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard4/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard5/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard6/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard7/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard8/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python k2t.py -file_name=data/shard9/keywords.txt -results_subfolder=finetunedgptmed_lr2e5_epoch2 -do_guarantee=True -n_generated_sentences=120
```
Possible hyperparameters are as follows:
```
-file_name: the location for each extracted keyword file. (currently, no change needed)
-results_subfolder: the name of the subfolder under 'results/' to which the generations saved. (currently, no change needed)
-n_generated_sentences: sentence length.
-do_guarantee: whether to guarantee appearance of given keywords in the generated text.
-top_p: nucleus sampling parameter
-weight: shift strenght $\lambda_0$.
-det_BS: deterministic beam search
-task: should not be changed.
```

## Step 5: Train real vs fake BERT Classifier

Go to train_fake_classifier.py.

Change line 15 data_dir with the dir 'processed_df_valid.csv' is saved on (e.g.  '/cluster/scratch/{eth_username}/nlp_lss_datasets).

Change line 40 results_path with the full path of your 'results/' folder (e.g. '/cluster/home/{eth_username}/conditioned_speech_gen/results').
Change line 41 folder_name with the name you entered to '-results_subfolder' in Step 4. (e.g.'finetunedgptmed_lr2e5_epoch2' )
Change line 42 experiment name with the name you find when you go to 'results/folder_name/your_experiment_settingsSENTENCES.txt' (e.g. 'Result_w_5.0_nBeams_1_nGenSent_128_nWordsPerSent_1_topP_0.9_WC_glove_maxSENTENCES.txt')

Create a temporary .py file, and run only once, and delete the temporary .py file: (data_dir is the same with line 15)
```
import pandas as pd
quality_fake_df = pd.DataFrame(columns=['speech','perplexity'])
quality_fake_df.to_csv(os.path.join(data_dir,'quality_fake_df.csv'),index=False)
```
This is your high quality synthetic dataset that will compile only the fake samples which tricked the BERT classifier across different experiments runs. Only run it once before frequent experimentation.

Then run on GPU with
```
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python train_fake_classifier.py
```

## Step 6: Evaluate Compiled High Quality Dataset with BERT Classifier

Go to train_quality_evaluator.py.

Change line 15 data_dir with the dir 'processed_df_valid.csv' is saved on (e.g.  '/cluster/scratch/{eth_username}/nlp_lss_datasets). If you have access to 'processed_df_valid.csv' and 'quality_fake_df.csv' files, put them under this directory.

Then run on GPU with
```
bsub -n 4 -W 23:59 -o euler_message -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceRTX2080Ti]" python train_quality_evaluator.py
```

Note: This repository borrows code from https://github.com/dapascual/K2T
