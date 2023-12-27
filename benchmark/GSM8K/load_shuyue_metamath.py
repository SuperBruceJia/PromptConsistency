# coding=utf-8

import jsonlines
from datasets import load_dataset


# dataset = load_dataset("shuyuej/MetaMathQA")
dataset = load_dataset("shuyuej/metamath_gsm8k")

print(dataset)
