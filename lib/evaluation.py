# coding=utf-8

import gc
import sys
import time
from collections import Counter

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
    :param data_path: dataset path
    :param file_path: save file path and file name
    """
    start_t = time.time()
    max_new_tokens = config.get("max_new_tokens")
    save_dir = config.get("save_dir")
    num_gpus = config.get("num_gpus")
    llama_path = config.get("llama_path")

    dataset = load_dataset("shuyuej/temporary_testing_data")
    dataset = dataset["test"]

    ids = dataset["id"]
    max_id = max(ids)
    instances = []
    acc = []
    for id in range(max_id):
        # Select all lines where 'id' is equal to id
        lines = dataset.filter(lambda example: example['id'] == id, batch_size=None)

        # Retrieved the original question
        ori_phrase = lines["original_question"][0]

        # Retrieved the paraphrased questions
        phrase = []
        for q in lines["paraphrased_question"]:
            phrase.append(q)

        pairs = []
        if len(phrase) >= 2:
            for i in range(len(phrase)):
                for j in range(i + 1, len(phrase)):
                    pairs.append([ori_phrase, phrase[i], phrase[j]])
                    pairs.append([ori_phrase, phrase[j], phrase[i]])

            for i in range(len(pairs)):
                pair = pairs[i]
                prompt = gsm8k_prompt(question=pair)
                instances.append(prompt)
        else:
            pairs = phrase
            pairs.append(ori_phrase)
            pairs.reverse()
            prompt = gsm8k_prompt(question=pairs)
            instances.append(prompt)

        # Get the label answer --> gsm8k_answers
        answer = lines["answer_detail"][0]
        temp_ans = answer.split('#### ')[1]
        temp_ans = int(temp_ans.replace(',', ''))

        responses = []
        stop_tokens = stop_token_list()
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
        llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.80)
        lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, save_dir + '/adapter')

        completions = llm.generate(instances, sampling_params)
        for output in completions:
            temp_gen = output.outputs[0].text
            responses.append(temp_gen)

        print('Successfully finished generating', len(instances), 'samples!')
        predictions = []
        for response_item in responses:
            pred = extract_number(response_item)
            predictions.append(pred)

        # Count occurrences of each element
        element_counts = Counter(predictions)

        # Find the most common element
        final_pred = element_counts.most_common(1)[0][0]
        if final_pred is not None:
            acc.append(float(final_pred) == float(temp_ans))
        else:
            acc.append(False)

        # Delete the llm object and free the memory
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")

    accuracy = sum(acc) / len(acc)
    end_t = time.time()
    elapsed_t = end_t - start_t
    print(f"Finished performance evaluation in {elapsed_t:.2f} seconds")
    print('Testing length:', len(acc), ' Accuracy:', accuracy)
