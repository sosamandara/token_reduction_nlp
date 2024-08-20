import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def evaluate_perplexity(model, encodings, times_to_repeat_stride):
  max_length = model.config.n_positions
  stride = 512
  seq_len = encodings.input_ids.size(1)

  log_probs = []
  device = 'cuda'

  for begin_loc in tqdm(range(0, stride*times_to_repeat_stride, stride)):
      end_loc = min(begin_loc + max_length, seq_len)
      input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

      # Process in batches
      for i in range(input_ids.size(1) - 1):
          with torch.no_grad():
              output = model(input_ids[:, :i + 1])
              logits = output.logits[:, -1, :]  # Get logits for the last token
              probs = torch.nn.functional.log_softmax(logits, dim=-1)
              target_token_id = input_ids[0, i + 1].item()
              log_prob = probs[0, target_token_id].item()
              log_probs.append(log_prob)

              if (i + 1) % 50 == 0:  # Print progress every 100 tokens
                  current_perplexity = np.exp(-np.mean(log_probs))
                  print(f"Current Perplexity: {current_perplexity}")

      if end_loc == seq_len:
          break
  return log_probs

def evaluate_perplexity_with_window(model, encodings, window_size, untill_sequence_index):
  stride = 1
  seq_len = encodings.input_ids.size(1)

  log_probs = []
  device = 'cuda'

  for begin_loc in tqdm(range(0, untill_sequence_index, stride)):
      end_loc = begin_loc + window_size
      if end_loc >= seq_len:
            break
      input_ids = encodings.input_ids[:, begin_loc:end_loc + stride].to(device)
      input_window = input_ids[:, :-1]  # Input window without the target token
      target_token_id = input_ids[0, -1].item()  # The target token to predict
      with torch.no_grad():
        output = model(input_window)
        logits = output.logits[:, -1, :]  # Get logits for the last token in the window
        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_prob = probs[0, target_token_id].item()
        log_probs.append(log_prob)

  avg_log_prob = np.mean(log_probs)
  perplexity = np.exp(-avg_log_prob)
  return perplexity, log_probs

def evaluate_and_save_average_perplexities_avg(models, model_names, encodings, window_sizes, output_dir, num_samples=10000):
    """
    Evaluate and save the average perplexity for a list of models.

    :param models: List of models to evaluate
    :param model_names: List of names corresponding to the models
    :param encodings: Encodings to use for evaluation
    :param window_sizes: List of window sizes to evaluate
    :param output_dir: Directory to save the results
    :param num_samples: Number of samples to evaluate (default: 10000)
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    average_perplexities = []

    for model, model_name in zip(models, model_names):
        total_perplexity = 0.0

        for window_size in window_sizes:
            print(f"Evaluating {model_name} with window size {window_size}")
            perplexity, _ = evaluate_perplexity_with_window(model, encodings, window_size, num_samples)
            total_perplexity += perplexity

        # Calculate the average perplexity across all window sizes
        average_perplexity = total_perplexity / len(window_sizes)
        average_perplexities.append(average_perplexity)

        print(f"Average perplexity for {model_name}: {average_perplexity}")

    # Create DataFrame for average perplexities
    df_average_perplexities = pd.DataFrame({
        'model_name': model_names,
        'average_perplexity': average_perplexities
    })

    # Save the DataFrame to a CSV file
    output_path = os.path.join(output_dir, 'average_perplexities.csv')
    df_average_perplexities.to_csv(output_path, index=False)

    print(f"Saved average perplexity results to {output_path}")
    print("Evaluation and saving of results completed.")

def evaluate_and_save_perplexities(models, model_names, encodings, window_sizes, output_dir, num_samples=10000):
    """
    Evaluate perplexity for a list of models and save the results to CSV files.

    :param models: List of models to evaluate
    :param model_names: List of names corresponding to the models
    :param encodings: Encodings to use for evaluation
    :param window_sizes: List of window sizes to evaluate
    :param output_dir: Directory to save the results
    :param num_samples: Number of samples to evaluate (default: 10000)
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for model, model_name in zip(models, model_names):
        perplexities = []
        log_probs_dict = {}

        for window_size in window_sizes:
            print(f"Evaluating {model_name} with window size {window_size}")
            perplexity, log_probs = evaluate_perplexity_with_window(model, encodings, window_size, num_samples)
            perplexities.append(perplexity)
            log_probs_dict[window_size] = log_probs

        # Create DataFrame for perplexities
        df_perplexities = pd.DataFrame({
            'window_size': window_sizes,
            'perplexity': perplexities
        })

        # Save the DataFrame to a CSV file
        output_path = os.path.join(output_dir, f'perplexities_{model_name}.csv')
        df_perplexities.to_csv(output_path, index=False)

        print(f"Saved perplexity results for {model_name} to {output_path}")

        # Save log probabilities for each window size
        for window_size in window_sizes:
            df_log_probs = pd.DataFrame({
                'log_probs': log_probs_dict[window_size]
            })
            log_probs_path = os.path.join(output_dir, f'log_probs_{model_name}_window_{window_size}.csv')
            df_log_probs.to_csv(log_probs_path, index=False)
            #print(f"Saved log probabilities for {model_name} with window size {window_size} to {log_probs_path}")

    print("Evaluation and saving of results completed.")


def recalculate_and_plot_perplexities_normal(window_sizes, model_names, output_dir):
    """
    Recalculate perplexities from log probabilities, save to CSV, and plot the results.

    :param window_sizes: List of window sizes to evaluate
    :param model_names: List of names corresponding to the models
    :param output_dir: Directory to save the results
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize dictionary to store the recalculated perplexities
    recalculated_perplexities = {model_name: [] for model_name in model_names}

    # Function to calculate perplexity from log probabilities
    def calculate_perplexity(log_probs):
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        return perplexity

    # Recalculate perplexities for each model and window size
    for model_name in model_names:
        for window_size in window_sizes:
            log_probs_filename = f'log_probs_{model_name}_window_{window_size}.csv'
            log_probs_path = os.path.join(output_dir, log_probs_filename)
            
            if not os.path.exists(log_probs_path):
                # Check alternative naming patterns
                if model_name == 'gpt2':
                    log_probs_filename = f'log_probs_gpt2_window_{window_size}.csv'
                else:
                    log_probs_filename = f'log_probs_custom_window_{window_size}.csv'
                
                log_probs_path = os.path.join(output_dir, log_probs_filename)
                
                if not os.path.exists(log_probs_path):
                    print(f"File not found: {log_probs_path}")
                    continue
            
            log_probs = pd.read_csv(log_probs_path)['log_probs'].tolist()
            perplexity = calculate_perplexity(log_probs)
            recalculated_perplexities[model_name].append(perplexity)

    # Save recalculated perplexities to CSV
    for model_name in model_names:
        df_recalculated = pd.DataFrame({
            'window_size': window_sizes,
            'perplexity': recalculated_perplexities[model_name]
        })
        output_path = os.path.join(output_dir, f'recalculated_perplexities_{model_name}.csv')
        df_recalculated.to_csv(output_path, index=False)
        #print(f"Saved recalculated perplexities for {model_name} to {output_path}")

    # Plotting the results
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Plot each model's perplexity
    for model_name in model_names:
        plt.plot(window_sizes, recalculated_perplexities[model_name], label=model_name.replace("custom", "Top"), marker='o', alpha=0.7)
    
    plt.xticks(window_sizes)  # Ensure x-axis has the correct window sizes as ticks
    plt.xlabel("Context Size (tokens)")
    plt.ylabel("Perplexity")
    plt.title("Perplexity Comparison of Models")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Save the plot to a file
    plot_path = os.path.join(output_dir, 'recalculated_perplexity_comparison_plot.png')
    plt.savefig(plot_path)
    plt.show()

    #print(f"Recalculated perplexities and plot saved to {output_dir}")


def recalculate_and_plot_perplexities(window_sizes, model_names, output_dir):
    """
    Recalculate perplexities from log probabilities, save to CSV, and plot the results.

    :param window_sizes: List of window sizes to evaluate
    :param model_names: List of names corresponding to the models
    :param output_dir: Directory to save the results
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize dictionary to store the recalculated perplexities
    recalculated_perplexities = {model_name: [] for model_name in model_names}

    # Function to calculate perplexity from log probabilities
    def calculate_perplexity(log_probs):
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        return perplexity

    # Recalculate perplexities for each model and window size
    for model_name in model_names:
        for window_size in window_sizes:
            log_probs_filename = f'log_probs_{model_name}_window_{window_size}.csv'
            log_probs_path = os.path.join(output_dir, log_probs_filename)
            
            if not os.path.exists(log_probs_path):
                # Check alternative naming patterns
                if model_name == 'gpt2':
                    log_probs_filename = f'log_probs_gpt2_window_{window_size}.csv'
                else:
                    log_probs_filename = f'log_probs_custom_window_{window_size}.csv'
                
                log_probs_path = os.path.join(output_dir, log_probs_filename)
                
                if not os.path.exists(log_probs_path):
                    print(f"File not found: {log_probs_path}")
                    continue
            
            log_probs = pd.read_csv(log_probs_path)['log_probs'].tolist()
            perplexity = calculate_perplexity(log_probs)
            recalculated_perplexities[model_name].append(perplexity)

    # Save recalculated perplexities to CSV
    for model_name in model_names:
        df_recalculated = pd.DataFrame({
            'window_size': window_sizes,
            'perplexity': recalculated_perplexities[model_name]
        })
        output_path = os.path.join(output_dir, f'recalculated_perplexities_{model_name}.csv')
        df_recalculated.to_csv(output_path, index=False)
        #print(f"Saved recalculated perplexities for {model_name} to {output_path}")

    # Plotting the results
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Plot with a log scale
    for model_name in model_names:
        plt.plot(window_sizes, recalculated_perplexities[model_name], label=model_name.replace("custom", "random"), marker='o')

    plt.yscale('log')
    plt.xticks(window_sizes)  # Ensure x-axis has the correct window sizes as ticks
    plt.xlabel("Context Size (tokens)")
    plt.ylabel("Perplexity")
    plt.title("Perplexity Comparison of Models")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, which="both", ls="--")

    # Annotate the plot
    for model_name in model_names:
        for i, window_size in enumerate(window_sizes):
            plt.annotate(f"{recalculated_perplexities[model_name][i]:.2f}",
                         (window_size, recalculated_perplexities[model_name][i]),
                         textcoords="offset points", xytext=(0,10), ha='center')

    # Save the plot to a file
    plot_path = os.path.join(output_dir, 'recalculated_perplexity_comparison_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()

    print(f"Recalculated perplexities and plot saved to {output_dir}")