import json
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Tuple, Union
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.flop_counter import FlopCounterMode
import os
current_dir = os.getcwd()
from modeling_topK_gpt2 import CustomGPT2LMHeadModel


def load_custom_model(model_name, config, k_percent, selection_method="top_k", layers_to_prune=None):
    return CustomGPT2LMHeadModel.from_pretrained(
        model_name,
        config=config,
        k_percent=k_percent,
        selection_method=selection_method,
        layers_to_prune=layers_to_prune
    ).to('cuda')

# Function to calculate FLOPs
def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    istrain = model.training
    model.eval()

    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops = flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops


# Function to generate text and calculate FLOPs
def generate_text(model, tokenizer, input_text, max_length=100):
    # tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

    # calculate flops
    flops = get_flops(model, input_ids)

    start = time.time()
    # generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    end = time.time()
    # decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text, flops, end-start


# Function to run generation on dataset and record results including FLOPs
def run_generation_on_dataset(dataset,
                              model: GPT2LMHeadModel,
                              tokenizer: GPT2Tokenizer,
                              num_examples: int,
                              lengths: list,
                              model_name: str,
                              folder_saved: str
                              ):
    results = []

    for idx in tqdm(range(num_examples), desc="Processing examples"):
        prefix = dataset["train"][idx]["text"]
        generated_text = dataset["train"][idx]["text"]

        for single_len in lengths:
            tot_time = 0
            total_flops = 0
            for _ in range(single_len):
                generated_text, flops, time = generate_text(model, tokenizer, generated_text, max_length=len(tokenizer(generated_text)["input_ids"])+1)
                total_flops += flops
                tot_time += time

            result = {
                "example_index": idx,
                "lengths": single_len,
                "prefix": prefix,
                "generated_text": generated_text[len(prefix):],
                "time_taken": tot_time,
                "flops": total_flops
            }

            results.append(result)

    file_name = f'results/{folder_saved}/generation_results_{model_name}.json'
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def calculate_averages(results, key):
    length_values = defaultdict(list)
    for result in results:
        length = result["lengths"]
        value = result[key]
        length_values[length].append(value)

    average_values = {length: sum(values) / len(values) for length, values in length_values.items()}
    return average_values


def plot_averages(custom_model_results, gpt2_model_results, key, ylabel, title, figsize=(10, 6)):
    custom_model_avg = calculate_averages(custom_model_results, key)
    gpt2_model_avg = calculate_averages(gpt2_model_results, key)

    sorted_lengths = sorted(custom_model_avg.keys())

    plt.figure(figsize=figsize)
    plt.plot(sorted_lengths, [custom_model_avg[length] for length in sorted_lengths], label='TopK Model', marker='o')
    plt.plot(sorted_lengths, [gpt2_model_avg[length] for length in sorted_lengths], label='GPT-2 Model', marker='o')
    plt.xlabel('Lengths')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
