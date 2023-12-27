# coding=utf-8

import gc
import sys
import time

import torch
import jsonlines

from vllm import LLM, SamplingParams
from vllm.model_executor.adapters import lora
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from utils.utils import (
    gsm8k_prompt,
    stop_token_list,
    extract_number,
)

MAX_INT = sys.maxsize


def gsm8k_test(config, file_path, data_path):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param data_path: dataset path
    :param file_path: save file path and file name
    """
    start_t = time.time()
    max_new_tokens = config.get("max_new_tokens")
    g_save_dir = config.get("g_save_dir")
    num_gpus = config.get("num_gpus")
    llama_path = config.get("llama_path")

    # Read the database and retrieve the label `gsm8k_answers`
    instances = []
    answers = []
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            # Get the prompt template + question --> gsm8k_ins
            temp_ins = gsm8k_prompt(question=item["question"], train=False)
            instances.append(temp_ins)

            # Get the label answer --> gsm8k_answers
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            answers.append(temp_ans)

    responses = []
    stop_tokens = stop_token_list()
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
    llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.85)
    lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, g_save_dir + '/adapter')

    completions = llm.generate(instances, sampling_params)
    for output in completions:
        temp_gen = output.outputs[0].text
        responses.append(temp_gen)

    print('Successfully finished generating', len(instances), 'samples!')
    acc = []
    invalid_out = []
    for idx, (instance_item, response_item, answer_item) in enumerate(zip(instances, responses, answers)):
        y_pred = extract_number(response_item)
        if y_pred is not None:
            acc.append(float(y_pred) == float(answer_item))
        else:
            acc.append(False)
            temp = {'question': instance_item, 'output': response_item, 'answer': answer_item}
            invalid_out.append(temp)

    accuracy = sum(acc) / len(acc)
    end_t = time.time()
    elapsed_t = end_t - start_t
    print(f"Finished performance evaluation in {elapsed_t:.2f} seconds")

    # Print the accuracy and the length of the invalid output
    print('Invalid output length:', len(invalid_out),
          ', Testing length:', len(acc),
          ', Accuracy:', accuracy)

    # Save the invalid output in a txt file
    file = open(file_path, 'w')
    file.write(str(invalid_out))
    file.close()
    print('Successfully saved the invalid output.')

    # Delete the llm object and free the memory
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")
