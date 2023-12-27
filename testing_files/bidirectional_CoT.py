#!/usr/bin/env python
# coding=utf-8

import yaml
import jsonlines

from vllm import LLM, SamplingParams
from huggingface_hub import login


def stop_token_list():
    stop_tokens = ['Question:',
                   'Question',
                   'Questions:',
                   'Questions',
                   'USER:',
                   'USER',
                   'USERS:',
                   'USERS',
                   'ASSISTANT:',
                   'ASSISTANT',
                   'ASSISTANTS:',
                   'ASSISTANTS',
                   'Instruction:',
                   'Instruction',
                   'Instructions:',
                   'Instructions',
                   'Response:',
                   'Response',
                   'Responses:',
                   'Responses',
                   'instruction',
                   'Below is an instruction that describes a task.',
                   ]

    return stop_tokens


def load_config():
    """Load parameters and path from the YAML file

    :return: The configuration info
    """
    fopen = open("config.yml")
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()

    return config


config = load_config()
test_path = config.get("test_path")
model_name = config.get("model_name")
num_gpus = config.get("num_gpus")
inference_max_len = config.get("inference_max_len")


def gsm8k_prompt(question):
    """The formatting prompts function for GSM8K database

    :param question: Question (task description)
    :param answer: Answer to the Question
    :return: The prompt of the GSM8K database
    """
    prompt = question + (" \nFirst, separate the question into sentences, indicating 'sentence 1' to 'sentence k'."
                         "\nSecond, think step by step and conduct the Chain-of-Thought Reasoning from left to right, "
                         "i.e., from “sentence 1” to “sentence k”, and give a numerical answer. \nThird, ignore the "
                         "Chain-of-Thought Reasoning from left to right path, think step by step, "
                         "and conduct Chain-of-Thought Reasoning from right to left, i.e., from 'sentence k' "
                         "to 'sentence 1', and give a numerical answer. \nFinally, consider the two reasoning paths "
                         "above and give a consistent final answer.")

    return prompt


stop_tokens = stop_token_list()
sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048, stop=stop_tokens)

login(token="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")
pipe = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.70)

data_path = test_path
with open(data_path, "r+", encoding="utf8") as f:
    for idx, item in enumerate(jsonlines.Reader(f)):
        prompt = gsm8k_prompt(item["question"])

        completions = pipe.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            print(generated_text, '\n\n\n\n')
