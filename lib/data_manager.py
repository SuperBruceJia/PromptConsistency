# coding=utf-8

import random
import copy
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, set_caching_enabled

from utils.utils import gsm8k_prompt

IGNORE_INDEX = -100


def dataset_maker(dataset):
    print("Start to make dataset!")
    new_dataset = []
    ids = dataset["id"]
    max_id = max(ids)

    for id in range(max_id):
        questions = []

        # Select all lines where 'id' is equal to id
        lines = dataset.filter(lambda example: example['id'] == id, batch_size=None)

        # Retrieved the paraphrased questions
        for q in lines["paraphrased_question"]:
            questions.append(q)
        questions.append(lines["original_question"][0])

        # Randomly select K items from the list
        num_q = random.randint(1, 12)
        try:
            selected_q = random.sample(questions, num_q)
        except BaseException:
            num_q = random.randint(1, len(questions))
            selected_q = random.sample(questions, num_q)

        answer = lines["answer_detail"][0]
        prompt = gsm8k_prompt(question=selected_q)
        new_dataset.append({"question": prompt, "answer": answer})

    random.shuffle(new_dataset)

    return new_dataset


def tokenize_fn(strings, tokenizer):
    """
    Tokenize a list of strings
    Args:
        strings: input sequence
        tokenizer: the defined tokenizer

    Returns: tokenization output
    (1) input_ids
    (2) labels
    (3) input_lens
    (4) labels_lens
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    """
    Preprocess the data by tokenizing
    Args:
        sources: questions
        targets: answers
        tokenizer: the defined tokenizer

    Returns: input_ids and target labels
    """
    examples = [
        s + t for s, t in zip(sources, targets)
    ]
    examples_tokenized, sources_tokenized = [
        tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self):
        super(SupervisedDataset, self).__init__()

        # Disable caching on a global scale
        set_caching_enabled(False)

        # Load the fine-tuning dataset
        # data = load_dataset("shuyuej/mathdata_consistency")
        data = load_dataset("shuyuej/GSM8K-Consistency")
        data = data["train"]
        data = dataset_maker(data)

        sources = [f"{example['question']}" for example in data]
        targets = [f"{example['answer']}" for example in data]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i]
        )

    def __getitem__(self, i):
        return dict(
            input_ids=self.sources[i],
            labels=self.targets[i]
        )


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def naive__call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances):
        sources = []
        targets = []

        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']

            sources.append(source)
            targets.append(target)

        data = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data['input_ids'], data['labels']

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def dataset_loader(tokenizer):
    """
    Make dataset and collator for supervised fine-tuning.
    Args:
        tokenizer: the defined Generator tokenizer
        discriminator: the defined discriminator for sequence classification
        pre_train: pre-training on the MetaMath dataset or not
        iterate: number of iterations

    Returns: fine-tuning dataset
    """
    dataset = SupervisedDataset()
    data_collator = DataCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
