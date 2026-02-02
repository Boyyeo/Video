import numpy as np 
import pandas as pd 
import torch
from load_physical_aiavdataset import load_physical_aiavdataset
from huggingface_hub import login
import os 
from tqdm import tqdm


login()
df = pd.read_parquet("clip_index.parquet")
df_test = df[df['split']=='test'] 
save_folder = "downloaded_clips"
os.makedirs(save_folder, exist_ok=True)
count = 0
for (index, row) in tqdm(df_test.iterrows()):
    if count > 5:
        break 
    if count % 2 == 0:
        print('{}/{} Progress Downloaded '.format(count, len(df_test)))
    clip_id = index 
    data = load_physical_aiavdataset(clip_id)
    #print(f"Data keys: {list(data.keys())}") 
    torch.save(data, f"{save_folder}/{str(count).zfill(6)}.pt")
    count += 1
