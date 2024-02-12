# coding=utf-8

import random
import jsonlines
from itertools import combinations

from datasets import load_dataset

from utils.utils import gsm8k_prompt, gsm8k_answer


def dataset_maker():
    # Load the fine-tuning dataset
    dataset = load_dataset("shuyuej/GSM8K-temp")
    dataset = dataset["train"]

    print("Start to make the GSM8K training dataset!")
    new_dataset = []
    ids = dataset["id"]
    max_id = max(ids)

    for id in range(max_id):
        ques = []
        answers = []

        # Select all lines where 'id' is equal to id
        lines = dataset.filter(lambda example: example['id'] == id, batch_size=None)

        # Retrieved the paraphrased questions
        for q in lines["paraphrased_question"]:
            ques.append(q)
        ques.append(lines["original_question"][0])
        ques = list(set(ques))
        random.shuffle(ques)

        # Retrieved the answer details
        for a in lines["answer_detail"]:
            # Find the index of "#### "
            start_index = a.find("#### ")

            # Find the index of "The answer is: "
            end_index = a.find("The answer is: ")

            # Remove the substring from "#### " to "The answer is: "
            a = a[:start_index] + a[end_index:]
            answers.append(a)

        answers = list(set(answers))
        print("answers: ", len(answers))
        # answer = lines["answer_detail"][0]

        # # Generator expression for all combinations from single element to all elements
        # all_comb = chain.from_iterable(combinations(ques, r) for r in range(1, len(ques) + 1))

        # Set the limit of combinations to 5
        max_combinations = 5

        # Generate combinations from single element to all
        all_comb = []
        for r in range(1, min(len(ques), max_combinations) + 1):
            current_comb = combinations(ques, r)
            current_comb = list(current_comb)

            number = 10
            if len(current_comb) > number:
                current_comb = random.sample(current_comb, number)

            all_comb.extend(current_comb)

        # Print the result
        for comb in all_comb:
            question = list(comb)
            prompt = gsm8k_prompt(question=question, inference=False)
            new_dataset.append({
                "question": prompt,
                "answer": gsm8k_answer(answers=answers, number=len(question))
            })

        print("Finished processing id:", id, "The length of the dataset is", len(new_dataset))

    # Load the fine-tuning dataset
    dataset = load_dataset("shuyuej/MATH-Consistency")
    dataset = dataset["train"]

    print("Start to make the MATH training dataset!")
    ids = dataset["id"]
    max_id = max(ids)

    for id in range(max_id):
        ques = []
        answers = []

        # Select all lines where 'id' is equal to id
        lines = dataset.filter(lambda example: example['id'] == id, batch_size=None)

        # Retrieved the paraphrased questions
        for q in lines["paraphrased_question"]:
            ques.append(q)
        ques.append(lines["original_question"][0])
        ques = list(set(ques))
        random.shuffle(ques)

        # Retrieved the answer details
        # answer = lines["answer_detail"][0]
        for a in lines["answer_detail"]:
            answers.append(a)
        answers = list(set(answers))
        print("answers: ", len(answers))

        # # Generator expression for all combinations from single element to all elements
        # all_comb = chain.from_iterable(combinations(ques, r) for r in range(1, len(ques) + 1))

        # Set the limit of combinations to 5
        max_combinations = 5

        # Generate combinations from single element to all
        all_comb = []
        for r in range(1, min(len(ques), max_combinations) + 1):
            current_comb = combinations(ques, r)
            current_comb = list(current_comb)

            number = 10
            if len(current_comb) > number:
                current_comb = random.sample(current_comb, number)

            all_comb.extend(current_comb)

        # Print the result
        for comb in all_comb:
            question = list(comb)
            prompt = gsm8k_prompt(question=question, inference=False)
            new_dataset.append({
                "question": prompt,
                "answer": gsm8k_answer(answers=answers, number=len(question))
            })

        print("Finished processing id:", id, "The length of the dataset is", len(new_dataset))

    # random.shuffle(new_dataset)
    print("The number of data in the dataset is", len(new_dataset), '\n\n')

    return new_dataset


if __name__ == "__main__":
    dataset = dataset_maker()

    # Save the modified data to a jsonl file
    output_file = 'gsm8k_and_math_training_fewer.jsonl'
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(dataset)
    print(f"Modified data saved to {output_file}")
