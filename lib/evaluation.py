# coding=utf-8

import gc
import sys
import time
import random
from collections import Counter
from itertools import permutations

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.model_executor.adapters import lora
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from utils.utils import (
    gsm8k_prompt,
    stop_token_list,
    extract_number,
)

MAX_INT = sys.maxsize


def gsm8k_test(config):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    """
    start_t = time.time()
    max_new_tokens = config.get("max_new_tokens")
    save_dir = config.get("save_dir")
    num_gpus = config.get("num_gpus")
    llama_path = config.get("llama_path")

    # Load dataset
    dataset = load_dataset("shuyuej/temporary_testing_data")
    dataset = dataset["test"]

    # Load LLM
    stop_tokens = stop_token_list()
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
    llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.80)
    lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, save_dir + '/adapter')

    # Start to prepare prompts
    ids = dataset["id"]
    max_id = max(ids)
    acc = []
    for id in range(max_id):
        prompts = []
        phrase = []

        # Select all lines where 'id' is equal to id
        lines = dataset.filter(lambda example: example['id'] == id, batch_size=None)

        # Retrieved the original question
        ori_phrase = lines["original_question"][0]

        # Retrieved the paraphrased questions
        for q in lines["paraphrased_question"]:
            phrase.append(q)

        num_q = 3
        if len(phrase) >= num_q:
            random.seed(0)
            selections = random.sample(phrase, num_q)
            selections.append(ori_phrase)

            random.seed(0)
            random.shuffle(selections)

            # Get all permutations of length num_q
            combinations = permutations(selections, num_q)

            # Get each combination
            for combo in combinations:
                prompt = gsm8k_prompt(question=list(combo))
                prompts.append(prompt)
        else:
            prompt = gsm8k_prompt(question=phrase)
            prompts.append(prompt)

        # pairs = []
        # if len(phrase) >= 2:
        #     for i in range(len(phrase)):
        #         for j in range(i + 1, len(phrase)):
        #             pairs.append([ori_phrase, phrase[i], phrase[j]])
        #             pairs.append([ori_phrase, phrase[j], phrase[i]])
        #
        #     for i in range(len(pairs)):
        #         pair = pairs[i]
        #         prompt = gsm8k_prompt(question=pair)
        #         prompts.append(prompt)
        #
        #     number = 64
        #     if len(prompts) >= number:
        #         random.seed(0)
        #         prompts = random.sample(prompts, number)
        #
        # else:
        #     pairs = phrase
        #     pairs.append(ori_phrase)
        #     pairs.reverse()
        #     prompt = gsm8k_prompt(question=pairs)
        #     prompts.append(prompt)

        # Get the label answer --> gsm8k_answers
        answer = lines["answer_detail"][0]
        ans = answer.split('#### ')[1]
        label = int(ans.replace(',', ''))

        # Generate results
        preds = []
        completions = llm.generate(prompts, sampling_params)
        for output in completions:
            gen = output.outputs[0].text
            pred = extract_number(gen)
            preds.append(pred)

        print('Testing ID:', id, ', successfully finished generating', len(prompts), 'samples!')

        # Count occurrences of each element
        counts = Counter(preds)

        # Find the most common element
        prediction = counts.most_common(1)[0][0]
        if prediction is not None:
            acc.append(float(prediction) == float(label))
        else:
            acc.append(False)

    # Calculate accuracy
    accuracy = sum(acc) / len(acc)
    end_t = time.time()
    elapsed_t = end_t - start_t
    print(f"Finished performance evaluation in {elapsed_t:.2f} seconds")
    print('Testing length:', len(acc), ' Accuracy:', accuracy)

    # Delete the llm object and free the memory
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory.\n")
