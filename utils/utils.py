# coding=utf-8

import re
import yaml
import random
from fraction import Fraction

import transformers


DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "<unk>"


def is_number(s):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param s:
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def extract_number(completion):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param completion: The model's generated response
    :return: The extracted answer number from the completion
    """
    text = completion.split('the final numerical answer is: ')

    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)

        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]

                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_special_token(tokenizer):
    """
    Add special tokens to the tokenizer
    """
    tokenizer.add_special_tokens(
        {
            "pad_token": DEFAULT_PAD_TOKEN,
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    return tokenizer


def stop_token_list():
    stop_tokens = [
        "Question:",
        "Question",
        "USER:",
        "USER",
        "ASSISTANT:",
        "ASSISTANT",
        "Instruction:",
        "Instruction",
        "Response:",
        "Response",
    ]

    return stop_tokens


def load_config():
    """Load parameters and path from the YAML file

    :return: The configuration info
    """
    fopen = open("config.yml")
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()

    return config


def print_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    # Retrieve a list of all named parameters in the model
    model_parameters = list(model.named_parameters())

    # Calculate the total number of parameters using a generator expression
    all_param = sum(p.numel() for _, p in model_parameters)

    # Calculate the total number of trainable parameters using a generator expression
    # that filters parameters which require gradients
    trainable_params = sum(p.numel() for _, p in model_parameters if p.requires_grad)

    # Print out the number of trainable parameters, total parameters,
    # and the percentage of parameters that are trainable
    # The percentage is formatted to two decimal places
    print(
        f"Trainable params: {trainable_params:,} | "
        f"All params: {all_param:,} | "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


class CustomStream:
    def __init__(self, filename, console_stream):
        self.filename = filename
        self.console_stream = console_stream

    def write(self, text):
        with open(self.filename, 'a') as file:
            file.write(text)
        self.console_stream.write(text)

    def flush(self):
        pass


def gsm8k_prompt(question, inference=True):
    """Prompt format for the GSM8K database

    :param question: Question (task description)
    :param train: Add a "Let's think step by step." during inference
    :return: prompt for the LLMs
    """
    prompt = ("Below are semantics-preserving instructions that describe a task. "
              "Write responses that appropriately completes these requests. "
              "At the end of these responses, please write your final numerical answer. ")

    for i, q in enumerate(question):
        prompt += "\n\n### Instruction "
        prompt += str(i + 1)
        prompt += ":\n"
        prompt += q

        if i < len(question) - 1:
            prompt += " "

    if inference:
        prompt += "\n\n### Response: Let's think step by step."
    else:
        prompt += "\n\n### Response:"

    return prompt


def gsm8k_answer(answers, number):
    ans = answers[0].split('The answer is: ')[1]

    response = ""
    for i in range(number):
        if i == 0:
            response += "Response to the instruction "
        else:
            response += "\n\nResponse to the instruction "

        response += str(i + 1)
        response += ":\n"
        response += random.sample(answers, 1)[0]
        response += " "

    response += "\n\nChecking all the answers above ("
    for i in range(number):
        response += "The answer to the instruction "
        response += str(i + 1)
        response += " is "
        response += ans

        if i < number - 1:
            response += ", "

    response += "), the final numerical answer is: "
    response += ans

    return response


def unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def model_saver(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
