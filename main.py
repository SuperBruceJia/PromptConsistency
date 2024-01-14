#!/usr/bin/env python
# coding=utf-8
#
# MIT License
#
# Copyright (c) 2023 Shuyue Jia
#
# DEPENDENCY ACKNOWLEDGMENT
# vLLM (Apache-2.0 license):
# https://github.com/vllm-project/vllm and https://github.com/troph-team/vllm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import logging

import yaml
import torch
from datetime import datetime
import transformers
from huggingface_hub import login

from lib.model_loader import model_initialize, trainer_loader
from lib.data_manager import dataset_loader
from lib.evaluation import (
    gsm8k_test_one,
    gsm8k_test_multiple,
)
from utils.utils import (
    CustomStream,
    load_config,
    model_saver,
)


def main(config):
    """Run the program"""
    # Retrieve the pathes of needed hyperparameters
    save_dir = config.get("save_dir")
    epochs = config.get("epochs")

    print("Initialize the model and tokenizer!")
    model, tokenizer = model_initialize(config)
    # model.save_pretrained(save_dir + '/adapter')

    # Performance evaluation on the testing set
    print("Evaluate the pretrained model's performance on the Testing Set")
    gsm8k_test_one(config=config)
    gsm8k_test_multiple(config=config)

    for iterate in range(epochs):
        print("Training iteration %s" % str(iterate))
        data = dataset_loader(tokenizer=tokenizer)
        trainer = trainer_loader(
            config,
            model=model,
            tokenizer=tokenizer,
            data_module=data,
            num_train_epochs=5
        )
        trainer.train()
        trainer.save_state()
        model_saver(trainer=trainer, output_dir=save_dir)
        # trainer.model.save_pretrained(save_dir + '/adapter')
        # print('Successfully save the pre-trained adapter of the Generator G!')
        tokenizer = trainer.tokenizer
        tokenizer.save_pretrained(save_dir)

        # Performance evaluation on the testing set
        print("Evaluate the model's performance on the Testing Set")
        gsm8k_test_one(config=config)
        gsm8k_test_multiple(config=config)


if __name__ == "__main__":
    # Hide Transformers warnings
    transformers.logging.set_verbosity_error()

    # Load the configuration
    config = load_config()
    result_dir = config.get("result_dir")
    hf_auth_token = config.get("hf_auth_token")
    hf_save_dir = config.get("hf_save_dir")

    # get the current working directory
    cwd = os.getcwd()
    login(token=hf_auth_token)  # Hugging Face Login
    os.environ['TRANSFORMERS_CACHE'] = hf_save_dir
    os.environ['HF_HOME'] = hf_save_dir
    os.environ['HF_DATASETS_CACHE'] = hf_save_dir

    # print output to the console
    print('\n\nThe current working directory is', cwd, '\n\n')

    # Check out the system assigned GPU id
    count = torch.cuda.device_count()
    print('There are', count, 'GPU/GPUs available!',
          'The devices are:', os.getenv("CUDA_VISIBLE_DEVICES"), '\n')

    # Get the current date and time
    time = datetime.now()

    # Create a subdirectory with the current date
    dir = os.path.join(result_dir, time.strftime("%Y-%m-%d"))
    os.makedirs(dir, exist_ok=True)

    # Create a log file with the exact time as the file name
    name = time.strftime("%H-%M-%S.log.txt")
    path = os.path.join(dir, name)

    # Configure the logging module to write to the log file
    logging.basicConfig(
        filename=path,
        level=logging.INFO,  # Adjust the log level as needed
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Redirect sys.stdout to the custom stream
    stream = CustomStream(path, sys.stdout)

    sys.stdout = stream
    print(yaml.dump(config, default_flow_style=False), '\n\n')
    main(config=config)
    sys.stdout = sys.__stdout__
