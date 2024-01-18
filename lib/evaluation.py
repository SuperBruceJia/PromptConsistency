# coding=utf-8

import gc
import sys
import time
import random
from collections import Counter
# from itertools import permutations
import jsonlines

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.model_executor.adapters import lora
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from utils.utils import (
    gsm8k_prompt,
    stop_token_list,
    extract_number,
    evaluation_augmentation,
)

MAX_INT = sys.maxsize


def gsm8k_test(config):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param data_path: dataset path
    :param file_path: save file path and file name
    """
    start_t = time.time()
    max_new_tokens = config.get("max_new_tokens")
    num_gpus = config.get("num_gpus")
    data_path = config.get("test_path")
    save_dir = config.get("save_dir")
    llama_path = config.get("llama_path")

    # Start the vLLM server
    stop_tokens = stop_token_list()
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
    llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.85)
    lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, save_dir + '/adapter')

    acc = []
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            ori_phrase = item["question"]

            phrase = []
            prompts = []
            num = 6
            for i in range(num):
                sen = evaluation_augmentation(sen=ori_phrase)
                phrase.append(sen)

            pairs = []
            for i in range(len(phrase)):
                for j in range(i + 1, len(phrase)):
                    pairs.append([ori_phrase, phrase[i], phrase[j]])
                    pairs.append([ori_phrase, phrase[j], phrase[i]])

            for pair in pairs:
                prompt = gsm8k_prompt(question=pair, train=False)
                prompts.append(prompt)

            # Get the label answer --> gsm8k_answers
            label = item['answer'].split('#### ')[1]
            label = int(label.replace(',', ''))

            # Generate results
            preds = []
            completions = llm.generate(prompts, sampling_params)
            for output in completions:
                gen = output.outputs[0].text
                pred = extract_number(gen)
                preds.append(pred)

            print('Testing ID:', idx, ', successfully finished generating', len(prompts), 'samples!')

            # Count occurrences of each element
            counts = Counter(preds)

            # Find the most common element
            prediction = counts.most_common(1)[0][0]
            if prediction is not None:
                acc.append(float(prediction) == float(label))
            else:
                acc.append(False)

    accuracy = sum(acc) / len(acc)
    print('Testing length:', len(acc), ' Accuracy:', accuracy)
    end_t = time.time()
    elapsed_t = end_t - start_t
    print(f"Finished performance evaluation in {elapsed_t:.2f} seconds")

    # Delete the llm object and free the memory
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")
