# coding=utf-8

import re

import jsonlines
from datasets import load_dataset, Features, Value


def clean_up(sentence):
    # Find all the locations of "<<"
    matches = [match.start() for match in re.finditer(r'<<', sentence)]

    for match in matches:
        # Get the left 20 characters of each "<<"
        left_chars = sentence[match-20:match]
        # Replace "x" or "X" to "*" if they are in the left 20 characters
        modified_chars = sentence[match-20:match].replace('x', '*').replace('X', '*')

        # Modify the original sentence
        if 'x' in left_chars or 'X' in left_chars:
            sentence = sentence.replace(left_chars, modified_chars)

    ##############################################################################################################

    # Define a pattern to match text between "<< and >>"
    pattern = r"<<(.*?)>>"

    # Use re.sub to replace matched patterns with an empty string
    sentence = re.sub(pattern, "", sentence)

    ##############################################################################################################
    # Find all occurrences of "*"
    asterisks = [i for i, char in enumerate(sentence) if char == '*']

    # Check and add spaces around "*"
    for index in reversed(asterisks):
        if index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] != ' ':
            sentence = sentence[:index] + ' ' + sentence[index] + ' ' + sentence[index + 1:]
        elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] == ' ':
            sentence = sentence[:index] + ' ' + sentence[index] + sentence[index + 1:]
        elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] == ' ' and sentence[index + 1] != ' ':
            sentence = sentence[:index] + sentence[index] + ' ' + sentence[index + 1:]

    ##############################################################################################################
    # # Find all occurrences of "/"
    # asterisks = [i for i, char in enumerate(sentence) if char == '/']
    #
    # # Check and add spaces around "/"
    # for index in reversed(asterisks):
    #     if index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] != ' ':
    #         sentence = sentence[:index] + ' ' + sentence[index] + ' ' + sentence[index + 1:]
    #     elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] == ' ':
    #         sentence = sentence[:index] + ' ' + sentence[index] + sentence[index + 1:]
    #     elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] == ' ' and sentence[index + 1] != ' ':
    #         sentence = sentence[:index] + sentence[index] + ' ' + sentence[index + 1:]

    ##############################################################################################################
    # Find all occurrences of "+"
    asterisks = [i for i, char in enumerate(sentence) if char == '+']

    # Check and add spaces around "+"
    for index in reversed(asterisks):
        if index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] != ' ':
            sentence = sentence[:index] + ' ' + sentence[index] + ' ' + sentence[index + 1:]
        elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] == ' ':
            sentence = sentence[:index] + ' ' + sentence[index] + sentence[index + 1:]
        elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] == ' ' and sentence[index + 1] != ' ':
            sentence = sentence[:index] + sentence[index] + ' ' + sentence[index + 1:]

    ##############################################################################################################
    # Find all occurrences of "-"
    asterisks = [i for i, char in enumerate(sentence) if char == '-']

    # Check and add spaces around "-"
    for index in reversed(asterisks):
        if index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] != ' ':
            sentence = sentence[:index] + ' ' + sentence[index] + ' ' + sentence[index + 1:]
        elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] == ' ':
            sentence = sentence[:index] + ' ' + sentence[index] + sentence[index + 1:]
        elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] == ' ' and sentence[index + 1] != ' ':
            sentence = sentence[:index] + sentence[index] + ' ' + sentence[index + 1:]

    ##############################################################################################################
    # Find all occurrences of "="
    asterisks = [i for i, char in enumerate(sentence) if char == '=']

    # Check and add spaces around "="
    for index in reversed(asterisks):
        if index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] != ' ':
            sentence = sentence[:index] + ' ' + sentence[index] + ' ' + sentence[index + 1:]
        elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] != ' ' and sentence[index + 1] == ' ':
            sentence = sentence[:index] + ' ' + sentence[index] + sentence[index + 1:]
        elif index > 0 and index < len(sentence) - 1 and sentence[index - 1] == ' ' and sentence[index + 1] != ' ':
            sentence = sentence[:index] + sentence[index] + ' ' + sentence[index + 1:]

    ##############################################################################################################
    # Find all occurrences of "."
    dots_locations = [match.start() for match in re.finditer(r'\.', sentence)]

    # Check and modify "." if the left side is space and the right side is a numerical number
    for dot_location in reversed(dots_locations):
        if sentence[dot_location - 1].isspace() and sentence[dot_location + 1].isdigit():
            sentence = sentence[:dot_location] + '0' + sentence[dot_location:]

    ##############################################################################################################
    # Check if there is a "." before "\n#### "
    if ".\n#### " not in sentence:
        # If not, add a "."
        sentence = sentence.replace("\n#### ", ".\n#### ")

    return sentence


# Retrieve the path of training and testing databases
context_feat = Features({"question": Value(dtype='string', id=None), "answer": Value(dtype='string', id=None)})
train_set = load_dataset('json', data_files='train.jsonl', split='train', features=context_feat)

data = []
for example in train_set:
    number = example['answer'].split('#### ')[1]
    number = int(number.replace(',', ''))
    append = "\nThe answer is: " + str(number)
    answer = example['answer'] + append
    answer = clean_up(sentence=answer)

    question = example['question']
    data.append({"question": question, "answer": answer})

# Save the modified data to a jsonl file
output_file = 'gsm8k_train.jsonl'
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(data)

print(f"Modified data saved to {output_file}")
