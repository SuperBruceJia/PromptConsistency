# # coding=utf-8
#
# import jsonlines
# import datasets
# from datasets import load_dataset
# import matplotlib.pyplot as plt
#
#
# # Load the dataset
# MetaMathQA = load_dataset("meta-math/MetaMathQA")
# MetaMathQA = MetaMathQA["train"]
#
# data = []
# originals = []
# ids = []
# for example in MetaMathQA:
#     type = example['type']
#
#     if type == "MATH_Rephrased" or type == "MATH_AnsAug":
#         ori_question = example['original_question']
#         if ori_question not in originals:
#             originals.append(ori_question)
#
#         ques_id = originals.index(ori_question)
#         ids.append(ques_id)
#         question = example['query']
#         answer = example['response']
#
#         data.append(
#             {"id": ques_id,
#              "original_question": ori_question,
#              "paraphrased_question": question,
#              "answer_detail": answer}
#         )
#
# print(datasets.Dataset.from_list(data))
#
# # Rank the lines by "id"
# data = sorted(data, key=lambda x: x["id"])
#
# # Create a histogram
# plt.hist(ids, bins=max(ids) + 1, edgecolor='black')
#
# # Add labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of List a')
#
# # Show the plot
# plt.show()
#
# # Save the modified data to a jsonl file
# output_file = 'math_consistency.jsonl'
# with jsonlines.open(output_file, 'w') as writer:
#     writer.write_all(data)
#
# print(f"Modified data saved to {output_file}")

# coding=utf-8

import jsonlines
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the dataset
MetaMathQA = load_dataset("meta-math/MetaMathQA")
MetaMathQA = MetaMathQA["train"]

data = []
originals = []
ids = []
for example in MetaMathQA:
    type = example['type']

    if type == "GSM_Rephrased" or type == "GSM_AnsAug":
        ori_question = example['original_question']
        if ori_question not in originals:
            originals.append(ori_question)

        ques_id = originals.index(ori_question)
        ids.append(ques_id)
        question = example['query']
        answer = example['response']

        data.append(
            {
                "id": ques_id,
                "original_question": ori_question,
                "paraphrased_question": question,
                "answer_detail": answer
            }
        )

print(datasets.Dataset.from_list(data))

# Rank the lines by "id"
data = sorted(data, key=lambda x: x["id"])

# Create a histogram
plt.hist(ids, bins=max(ids) + 1, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of List a')

# Show the plot
plt.show()

# Save the modified data to a jsonl file
output_file = 'gsm8k_consistency.jsonl'
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(data)

print(f"Modified data saved to {output_file}")
