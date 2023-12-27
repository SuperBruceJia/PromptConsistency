#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from datasets import load_dataset

# Load datasets
# train_data = load_dataset('json', data_files='train.jsonl', split=None)
# test_data = load_dataset('json', data_files='test.jsonl', split=None)

# from datasets import load_dataset
# from trl import SFTTrainer
#
# dataset = load_dataset("imdb", split="train")
# print(dataset)

# import re
#
# from utils.utils import gsm8k_prompts_format
#
# question = ("Janet\u2019s ducks lay 16 eggs per day. "
#             "She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
#             "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
#             "How much in dollars does she make every day at the farmers' market?")
# answer = ("Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day "
#           "at the farmer\u2019s market.\n#### 18")
#
# prompt = gsm8k_prompts_format(question, answer)
# output = prompt.split("###")[-1].strip()
# match = re.search(r'[\-+]?\d*[\.,/]?\d+', output)
# print(match)


# # Read the database and retrieve the label `gsm8k_answers`
# import jsonlines
# from utils.utils import extract_answer_number
# gsm8k_ins = []
# gsm8k_answers = []
#
# problem_prompt = (
#     "Below is an instruction that describes a task. "
#     "Write a response that appropriately completes the request.\n\n"
#     "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
# )
#
# data_path = "test.jsonl"
# with open(data_path, "r+", encoding="utf8") as f:
#     for idx, item in enumerate(jsonlines.Reader(f)):
#         temp_instr = problem_prompt.format(instruction=item["question"])
#         # print(temp_instr, '\n\n\n')
#
#         gsm8k_ins.append(temp_instr)
#         # print(extract_answer_number(completion=item['answer']))
#         temp_ans = item['answer'].split('#### ')[1]
#         temp_ans = int(temp_ans.replace(',', ''))
#         # print(temp_ans, '\n\n\n')
#         gsm8k_answers.append(temp_ans)
#
# # gsm8k_ins = gsm8k_ins[start:end]
# # gsm8k_answers = gsm8k_answers[start:end]
# print('lenght ====', len(gsm8k_ins))
