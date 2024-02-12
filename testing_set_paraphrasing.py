#!/usr/bin/env python
# coding=utf-8

import jsonlines
import threading

from utils.utils import (
    load_config,
    evaluation_augmentation,
)


class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        super().__init__(target=target, args=args)
        self._result = None

    def run(self):
        if self._target is not None:
            self._result = self._target(*self._args, **self._kwargs)

    def get_result(self):
        return self._result


def func(original_phrase):
    paraphrase = evaluation_augmentation(sen=original_phrase)

    return paraphrase


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

            # Create threads
            threads = []
            for _ in range(10):
                thread = MyThread(target=func, args=(ori_phrase,))
                threads.append(thread)

            # Start threads
            for thread in threads:
                thread.start()

            # Join threads and collect return values
            results = []
            for thread in threads:
                thread.join()
                results.append(thread.get_result())

            # Append the augmented phrases
            for i, paraphrase in enumerate(results):
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
    output_file = 'gsm8k_testing_set_aug_with_postprocessing.jsonl'
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(testing_aug)
    print(f"Modified data saved to {output_file}")


if __name__ == "__main__":
    # Load the configuration
    config = load_config()
    test_path = config.get("test_path")
    generate_data(test_path=test_path)


class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        super().__init__(target=target, args=args)
        self._result = None

    def run(self):
        if self._target is not None:
            self._result = self._target(*self._args, **self._kwargs)

    def get_result(self):
        return self._result


def func(original_phrase):
    paraphrase = evaluation_augmentation(sen=original_phrase)

    return paraphrase


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

            # Create threads
            threads = []
            for _ in range(10):
                thread = MyThread(target=func, args=(ori_phrase,))
                threads.append(thread)

            # Start threads
            for thread in threads:
                thread.start()

            # Join threads and collect return values
            results = []
            for thread in threads:
                thread.join()
                results.append(thread.get_result())

            # Append the augmented phrases
            for i, paraphrase in enumerate(results):
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
    output_file = 'gsm8k_testing_set_aug_with_postprocessing.jsonl'
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(testing_aug)
    print(f"Modified data saved to {output_file}")


if __name__ == "__main__":
    # Load the configuration
    config = load_config()
    test_path = config.get("test_path")
    generate_data(test_path=test_path)


# #!/usr/bin/env python
# # coding=utf-8
#
# import jsonlines
#
# from utils.utils import (
#     load_config,
#     evaluation_augmentation,
# )
#
#
# def generate_data(test_path):
#     """
#     Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
#     :param data_path: dataset path
#     :param file_path: save file path and file name
#     """
#
#     testing_aug = []
#     id = []
#     original_question = []
#     paraphrased_question = []
#     answer_detail = []
#
#     with open(test_path, "r+", encoding="utf8") as f:
#         for idx, item in enumerate(jsonlines.Reader(f)):
#             ori_phrase = item["question"]
#
#             num = 10
#             for i in range(num):
#                 paraphrase = evaluation_augmentation(sen=ori_phrase)
#
#                 # Check if the sentence ends with a period
#                 if paraphrase.endswith('.'):
#                     pass
#                 else:
#                     # Add a period if there is no period
#                     paraphrase += "."
#
#                 # Check if the first character is lowercase
#                 if paraphrase[0].islower():
#                     # Convert the first character to uppercase
#                     paraphrase = paraphrase.capitalize()
#
#                 id.append(idx)
#                 original_question.append(ori_phrase)
#                 paraphrased_question.append(paraphrase)
#                 answer_detail.append(item['answer'])
#
#             print("Finished processing the ID: ", idx)
#
#     for i in range(len(id)):
#         testing_aug.append(
#             {
#                 "id": id[i],
#                 "original_question": original_question[i],
#                 "paraphrased_question": paraphrased_question[i],
#                 "answer_detail": answer_detail[i],
#             }
#         )
#
#     # Save the modified data to a jsonl file
#     output_file = 'gsm8k_testing_set_aug_with_postprocessing.jsonl'
#     with jsonlines.open(output_file, 'w') as writer:
#         writer.write_all(testing_aug)
#     print(f"Modified data saved to {output_file}")
#
#
# if __name__ == "__main__":
#     # Load the configuration
#     config = load_config()
#     test_path = config.get("test_path")
#     generate_data(test_path=test_path)
