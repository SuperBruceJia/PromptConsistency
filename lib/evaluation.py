# coding=utf-8

import gc
import sys
import time
import random
from collections import Counter

import torch
import jsonlines
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


def gsm8k_test(config, file_path, data_path):
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

    # Read the database and retrieve the label `gsm8k_answers`
    instances = []
    answers = []
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            # Get the prompt template + question --> gsm8k_ins
            ins = item["question"]
            if type(ins) is not list:
                ins = [ins]
            temp_ins = gsm8k_prompt(question=ins)
            instances.append(temp_ins)

            # Get the label answer --> gsm8k_answers
            temp_ans = item["answer"].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            answers.append(temp_ans)

    responses = []
    stop_tokens = stop_token_list()
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
    llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.85)
    lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, save_dir + '/adapter')

    completions = llm.generate(instances, sampling_params)
    for i, output in enumerate(completions):
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
    print('Invalid output length:', len(invalid_out), ', Testing length:', len(acc), ', Accuracy:', accuracy)

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


# def gsm8k_test(config):
#     """
#     Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
#     :param data_path: dataset path
#     :param file_path: save file path and file name
#     """
#     start_t = time.time()
#     max_new_tokens = config.get("max_new_tokens")
#     save_dir = config.get("save_dir")
#     num_gpus = config.get("num_gpus")
#     llama_path = config.get("llama_path")
#
#     # Load dataset
#     dataset = load_dataset("shuyuej/temporary_testing_data")
#     dataset = dataset["test"]
#
#     # Load LLM
#     stop_tokens = stop_token_list()
#     sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
#     llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.80)
#     lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, save_dir + '/adapter')
#
#     ids = dataset["id"]
#     max_id = max(ids)
#     acc = []
#     for id in range(max_id):
#         prompts = []
#         responses = []
#         phrase = []
#
#         # Select all lines where 'id' is equal to id
#         lines = dataset.filter(lambda example: example['id'] == id, batch_size=None)
#
#         # Retrieved the original question
#         ori_phrase = lines["original_question"][0]
#
#         # Retrieved the paraphrased questions
#         for q in lines["paraphrased_question"]:
#             phrase.append(q)
#         # phrase.append(ori_phrase)
#
#         # random.seed(0)
#         # random.shuffle(phrase)
#         # random.seed(0)
#         # selected_q = random.sample(phrase, 3)
#
#         pairs = []
#         if len(phrase) >= 2:
#             for i in range(len(phrase)):
#                 for j in range(i + 1, len(phrase)):
#                     pairs.append([ori_phrase, phrase[i], phrase[j]])
#                     pairs.append([ori_phrase, phrase[j], phrase[i]])
#
#             for i in range(len(pairs)):
#                 pair = pairs[i]
#                 prompt = gsm8k_prompt(question=pair)
#                 prompts.append(prompt)
#
#             number = 64
#             if len(prompts) >= number:
#                 random.seed(0)
#                 prompts = random.sample(prompts, number)
#
#         else:
#             pairs = phrase
#             pairs.append(ori_phrase)
#             pairs.reverse()
#             prompt = gsm8k_prompt(question=pairs)
#             prompts.append(prompt)
#
#         # Get the label answer --> gsm8k_answers
#         answer = lines["answer_detail"][0]
#         ans = answer.split('#### ')[1]
#         label = int(ans.replace(',', ''))
#
#         completions = llm.generate(prompts, sampling_params)
#         for output in completions:
#             gen = output.outputs[0].text
#             responses.append(gen)
#
#         print('Regarding testing sample ID:', id, ', successfully finished generating', len(prompts), 'samples!')
#         preds = []
#         for response_item in responses:
#             pred = extract_number(response_item)
#             preds.append(pred)
#
#         # Count occurrences of each element
#         counts = Counter(preds)
#
#         # Find the most common element
#         prediction = counts.most_common(1)[0][0]
#         if prediction is not None:
#             acc.append(float(prediction) == float(label))
#         else:
#             acc.append(False)
#
#     # Calculate accuracy
#     accuracy = sum(acc) / len(acc)
#     end_t = time.time()
#     elapsed_t = end_t - start_t
#     print(f"Finished performance evaluation in {elapsed_t:.2f} seconds")
#     print('Testing length:', len(acc), ' Accuracy:', accuracy)
#
#     # Delete the llm object and free the memory
#     destroy_model_parallel()
#     del llm
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.distributed.destroy_process_group()
#     print("Successfully delete the llm pipeline and free the GPU memory.\n")
