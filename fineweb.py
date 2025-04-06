import os
import multiprocessing as mp
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
#-------------------------------------------------------
local_dir = "edu_fineweb108"
remote_name = "sample-10BT"
shard_size = int(1e8) #100 M tokens per shard ,total of 100 shards

#create the cache the local directory if it didn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the  dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu",name=remote_name,split="train")
 #init the tokenizer
enc = GPT2Tokenizer.from_pretrained('gpt2')
eot = enc._sep_token['<|endoftext|>']
def tokenize(doc):
    # tokenizes a single document and return a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0<=tokens_np).all() and (tokens_np < 2**16).all(),"tokens directory too large"
    tokens_np_unit16 = tokens_np.astype(np.uint16)
    return tokens_np_unit16

def write_datafile(filename,tokens_np):
    #write a numpy array of uint16 tokens to a binary file
    with open(filename,"wb") as f:
        f.write(tokens_np.tobytes())


#tokenize all documents and write output shards,each of shards_size tokens
nprocs = max(1,os.cpu_count()//2)
with  mp.Pool(nprocs) as pool:
    shard_index = 0
    #prelocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,),dtype = np.uint16)
    token_count = 0
    progress_bar = None
    for  tokens in pool.imap(tokenize,fw,chunksize=16):
        #is there enough space in space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            #simply append tokens to current shard:
            all_tokens_np[shard_index:shard_index+len(tokens)] = tokens
            token_count += len(tokens)
             #update _progress bar
            if progress_bar is None:
                progress_bar = tqdm(total = shard_size,unit = 'tokens',desc = f'Shard{shard_index:06d}')
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR,f"edufineweb_{split}_{shard_index:06d}")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[shard_index:shard_index+remainder] = tokens[:remainder]
            write_datafile(filename,all_tokens_np)
            shard_index += 1
            progress_bar = None
            #populate teh next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR,f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename,all_tokens_np[:token_count])
