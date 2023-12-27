#!/usr/bin/env python
# coding=utf-8

import os
import numpy as np
import jsonlines

from lib.loader import pre_trained_model_loader
from utils.utils import load_config, gsm8k_prompts_format

# Set the main working directory
os.chdir("../")

# Load the configurations
config = load_config()
train_path = config.get("train_path")
test_path = config.get("test_path")

# Load the tokenizer
_, _, tokenizer = pre_trained_model_loader(config=config)

# Choose the data path
data_path = train_path

# wo: without
# p: prompt
# q: question
# a: answer
num_a = []
num_wo_a = []
num_with_a = []
num_with_p_wo_a = []
num_with_p_with_a = []
with open(data_path, "r+", encoding="utf8") as f:
    for idx, item in enumerate(jsonlines.Reader(f)):
        question = item["question"]
        answer = item["answer"]

        tokens_q = tokenizer.tokenize(question)
        tokens_a = tokenizer.tokenize(answer)

        p_wo_a = gsm8k_prompts_format(question=question, answer=answer, inference=True)
        p_with_a = gsm8k_prompts_format(question=question, answer=answer, inference=False)

        tokens_p_wo_a = tokenizer.tokenize(p_wo_a)
        tokens_p_with_a = tokenizer.tokenize(p_with_a)

        # Number of tokens of question, i.e., without answer and without prompt
        num_wo_a.append(len(tokens_q))

        # Number of tokens of answer
        num_a.append(len(tokens_a))

        # Number of tokens with answer but without prompt
        num_with_a.append(len(tokens_q) + len(tokens_a))

        # Number of tokens with prompt but without answer
        num_with_p_wo_a.append(len(tokens_p_wo_a))

        # Number of tokens with prompt and with answer
        num_with_p_with_a.append(len(tokens_p_with_a))

    print('The number of data that has been evaluated',
          len(num_wo_a), len(num_a), len(num_with_a),
          len(num_with_p_wo_a), len(num_with_p_with_a), '\n\n')

    print('----------Number of tokens without answer and without prompt----------\n')
    print('Max: ', np.max(num_wo_a), 'Min: ', np.min(num_wo_a),
          'Mean: ', np.mean(num_wo_a), 'Median: ', np.median(num_wo_a),
          'STD: ', np.std(num_wo_a), '\n\n\n\n')

    print('----------Number of tokens of answer----------\n')
    print('Max: ', np.max(num_a), 'Min: ', np.min(num_a),
          'Mean: ', np.mean(num_a), 'Median: ', np.median(num_a),
          'STD: ', np.std(num_a), '\n\n\n\n')

    print('----------Number of tokens with answer but without prompt----------\n')
    print('Max: ', np.max(num_with_a), 'Min: ', np.min(num_with_a),
          'Mean: ', np.mean(num_with_a), 'Median: ', np.median(num_with_a),
          'STD: ', np.std(num_with_a), '\n\n\n\n')

    print('----------Number of tokens with prompt but without answer----------\n')
    print('Max: ', np.max(num_with_p_wo_a), 'Min: ', np.min(num_with_p_wo_a),
          'Mean: ', np.mean(num_with_p_wo_a), 'Median: ', np.median(num_with_p_wo_a),
          'STD: ', np.std(num_with_p_wo_a), '\n\n\n\n')

    print('----------Number of tokens with prompt and with answer----------\n')
    print('Max: ', np.max(num_with_p_with_a), 'Min: ', np.min(num_with_p_with_a),
          'Mean: ', np.mean(num_with_p_with_a), 'Median: ', np.median(num_with_p_with_a),
          'STD: ', np.std(num_with_p_with_a), '\n\n\n\n')
