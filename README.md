# Token Reduction in NLP

This repository implements token reduction techniques for Natural Language Processing (NLP) tasks, aiming to enhance model efficiency by dynamically pruning tokens based on attention scores.

The aggregation mechanism tried for now is the average.

This project forms part of a thesis exploring advanced NLP methods. The customized model is built by modifying the [GPT-2 model](https://github.com/openai/gpt-2).

## Brief Introduction
The aim of this project is to optimize transformer models by selectively pruning tokens that contribute less to the overall attention mechanism. This approach speeds up inference but also reduces computational overhead hopefully without impacting model performance so drastically.

Why this study can be helpful:

- *Efficiency Improvements*: Reducing the number of tokens processed by the model can significantly lower computation time and resources.
- *Maintaining Performance:* Despite the reduction in tokens, the model have a BERTScore value of ~0.8 similar to the one generated by the gpt2 model.


### Structure of the repo

    .
    ├── models/
    │   └── modeling_topK_gpt2.py
    ├── results/
    │   ├── generation_results_custom_model_50_with_flops.json
    │   └── generation_results_model_gpt_2_50_with_flops.json
    ├── LICENSE
    ├── README.md
    ├── Test_reduction.ipynb
    ├── functions.py
    └── retrieve_metrics_and_texts.ipynb

The notebook *retrieve_metrics_and_texts.ipynb* is runnable in Colab.

# The Data Set
The dataset used for experiments is the [ag_news](https://paperswithcode.com/dataset/ag-news).

# Experiments
Experiments were conducted to compare the performance and efficiency of models with and without token reduction. The configurations for the experiments and their results are provided in the results folder as JSON files.
The mask used in the custom model is:

[0.0, 0.0, 0.2, 0.2, 0.2, 0.15, 0.2, 0.2, 0.05, 0.05, 0.0, 0.05].

Further analysis will be conducted.


# Used technologies

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
 ![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
 ![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
