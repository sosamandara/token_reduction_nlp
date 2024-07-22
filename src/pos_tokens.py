import sys
import os
import warnings
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
import spacy
from spacy.tokens import Doc
# Suppress specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")

# Get the current working directory
current_dir = os.getcwd()

# Assuming the notebooks are in the notebooks directory and executed from there
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the models and src directories to the Python path
models_dir = os.path.join(project_root, 'models')
src_dir = os.path.join(project_root, 'src')

sys.path.append(models_dir)
sys.path.append(src_dir)

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModel
from modeling_topK_gpt2 import CustomGPT2LMHeadModel
import json
import matplotlib.pyplot as plt
from functions import run_generation_on_dataset, plot_averages, load_custom_model
import numpy as np
import pandas as pd

mask_30 = [0.0, 0.0, 0.0, 0.0, 0.72, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
model_gpt = GPT2LMHeadModel.from_pretrained(model_name, config=config).to('cuda')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Layers to prune
layers_to_prune = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#model_id = f"top_K_90"
top_K_30 =  load_custom_model(model_name, config, mask_30, selection_method="top_k", layers_to_prune=layers_to_prune)

# Function to generate text and calculate FLOPs
def generate_text(model, tokenizer, input_text, max_length=100):
    # tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
    
    # generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # decode the generated text
    #print(input_ids[model.indices[0,:]])
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    indices = model.indices

    removed_tokens = []
    for idx in indices[0, :]:
        removed_tokens.append(tokenizer.decode(input_ids[0, idx].item(), skip_special_tokens=True).strip())
    return generated_text, model.indices, removed_tokens, input_ids[0].tolist()

# POS tagging function
def get_pos_tags(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def normalize_token(token):
    # Remove leading 'Ġ' for consistency
    return token.replace('Ġ', '').lower()


def run_generation_on_dataset(dataset, model, tokenizer, num_examples, lengths, model_name, output_dir):
    results = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    token_counts = defaultdict(int)
    token_appearance_counts = defaultdict(int)
    pos_counts = defaultdict(int)
    pos_removal_counts = defaultdict(int)

    for idx in tqdm(range(num_examples), desc="Processing examples"):
        prefix = dataset["train"][idx]["text"]
        generated_text = dataset["train"][idx]["text"]

        all_removed_tokens = []
        all_input_ids = []

        for single_len in lengths:
            try:
                generated_text, indices, removed_tokens, input_ids = generate_text(model, tokenizer, generated_text, max_length=1024)
                all_removed_tokens.extend(removed_tokens)
                all_input_ids.extend(input_ids)
            except Exception as e:
                #print(f"Error processing example {idx}: {e}")
                continue

        result = {
            "example_index": idx,
            "prefix": prefix,
            "removed_tokens": all_removed_tokens,
            "removed_tokens_count": {token: all_removed_tokens.count(token) for token in set(all_removed_tokens)},
            "model_name": model_name
        }

        results.append(result)

        for token in all_removed_tokens:
            normalized_token = normalize_token(token)
            token_counts[normalized_token] += 1
        
        # Count token appearances
        tokens = tokenizer.convert_ids_to_tokens(all_input_ids)
        for token in tokens:
            normalized_token = normalize_token(token)
            token_appearance_counts[normalized_token] += 1

        # POS tagging and counting
        pos_tags = get_pos_tags(generated_text)
        for token, pos in pos_tags:
            normalized_token = normalize_token(token)
            pos_counts[pos] += 1
            if normalized_token in token_counts:
                pos_removal_counts[pos] += 1

    file_name = os.path.join(output_dir, f'token_removal_results_{model_name}.json')
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)

    return results, token_counts, token_appearance_counts, pos_counts, pos_removal_counts

# Specify a new cache directory
#os.environ['HF_DATASETS_CACHE'] = 'D:\cache'
# Load the ag_news dataset
dataset = load_dataset("imdb")

num_examples =  len(dataset["train"]) # Adjust as needed
lengths = [1]  # Adjust as needed
output_dir = "./output"
results, token_counts, token_appearance_counts, pos_counts, pos_removal_counts = run_generation_on_dataset(dataset, top_K_30, tokenizer, num_examples, lengths, model_name, output_dir)

# Create a list of tuples for the CSV
# Debug: Print token counts before creating CSV
#print("Token counts (removal):")
#for token, count in token_counts.items():
#    print(f"'{token}': {count}")
#
#print("\nToken counts (appearance):")
#for token, count in token_appearance_counts.items():
#    print(f"'{token}': {count}")

# Create a list of tuples for the CSV
data = []
for token, appearance_count in token_appearance_counts.items():
    #print("-------", token, "------", appearance_count, "------", token_counts.get(token, 0))
    removed_count = token_counts.get(token, 0)
    relative_removal = removed_count / appearance_count if appearance_count > 0 else 0
    data.append((token, removed_count, appearance_count, relative_removal))

# Sort data by removed_count in descending order
data = sorted(data, key=lambda x: x[1], reverse=True)


# Create a DataFrame and save as CSV
df = pd.DataFrame(data, columns=["token", "removed_token_count", "token_appearance_count", "relative_removal"])
df.to_csv(os.path.join(output_dir, f'token_removal_statistics_{model_name}.csv'), index=False)

# Create POS DataFrame and save as CSV
pos_data = []
for pos, count in pos_counts.items():
    removal_count = pos_removal_counts.get(pos, 0)
    relative_removal = removal_count / count if count > 0 else 0
    pos_data.append((pos, removal_count, count, relative_removal))

pos_df = pd.DataFrame(pos_data, columns=["pos", "removed_token_count", "token_appearance_count", "relative_removal"])
pos_df.to_csv(os.path.join(output_dir, f'pos_removal_statistics_{model_name}.csv'), index=False)

print("Results saved to the output directory.")

# Plot POS removal statistics
import seaborn as sns

# Plot POS removal counts
plt.figure(figsize=(14, 7))
sns.barplot(x='removed_token_count', y='pos', data=pos_df, palette='viridis')
plt.xlabel('Removed Token Count')
plt.ylabel('POS')
plt.title('POS Removal Counts')
plt.show()

# Plot POS relative removal
plt.figure(figsize=(14, 7))
sns.barplot(x='relative_removal', y='pos', data=pos_df, palette='magma')
plt.xlabel('Relative Removal')
plt.ylabel('POS')
plt.title('POS Relative Removal')
plt.show()
