# coding=utf-8

import jsonlines
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("meta-math/MetaMathQA")
dataset = dataset["train"]

data = []
for example in dataset:
    question = example['query']
    answer = example['response']

    data.append({"question": question, "answer": answer})

# Save the modified data to a jsonl file
output_file = 'MetaMathQA.jsonl'
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(data)

print(f"Modified data saved to {output_file}")
