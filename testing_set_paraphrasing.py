#!/usr/bin/env python
# coding=utf-8

import sys
import jsonlines

from utils.utils import (
    load_config,
    evaluation_augmentation,
)

MAX_INT = sys.maxsize


def generate_data(test_path):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param data_path: dataset path
    :param file_path: save file path and file name
    """

    testing_aug = []
    id = []
    original_question = []
    paraphrased_question = []
    answer_detail = []

    with open(test_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            ori_phrase = item["question"]

            num = 10
            for i in range(num):
                paraphrase = evaluation_augmentation(sen=ori_phrase)

                id.append(idx)
                original_question.append(ori_phrase)
                paraphrased_question.append(paraphrase)
                answer_detail.append(item['answer'])

            print("Finished processing the ID: ", idx)

    for i in range(len(id)):
        testing_aug.append(
            {
                "id": id[i],
                "original_question": original_question[i],
                "paraphrased_question": paraphrased_question[i],
                "answer_detail": answer_detail[i],
            }
        )

    # Save the modified data to a jsonl file
    output_file = 'gsm8k_testing_set_aug.jsonl'
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(testing_aug)
    print(f"Modified data saved to {output_file}")


if __name__ == "__main__":
    # Load the configuration
    config = load_config()
    test_path = config.get("test_path")
    generate_data(test_path=test_path)
