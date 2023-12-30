# Semantics-preserving Prompt Improves Self-consistency of Language Models
[![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](CODE_LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](DATA_LICENSE)
[![Model Weight License](https://img.shields.io/badge/Model%20Weights%20License-LLaMA2-yellow)](IntuitLLMProject/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## 🖼️ General Framework
![Framework.png](images/Framework.png)

## 📝 Preliminaries
### 1. Environment Setting-up
```python
CUDA: 11.3
Python: 3.9.4
PyTorch: 1.12.0

---

tensorboard
jsonlines
fraction
protobuf==3.20.0
PyYAML
accelerate==0.21.0
peft==0.4.0
typing_extensions==4.7.1
transformers==4.31.0
trl==0.4.7
bitsandbytes==0.39.0
googletrans==3.1.0a0
nltk
sentencepiece

---

# The following are the server setting-up codes.
# echo "Loading cuda/11.3"
# module load cuda/11.3
# 
# echo "Loading python3/3.8.10"
# module load python3/3.8.10
# 
# echo "Loading pytorch/1.12.0"
# module load pytorch/1.12.0
# 
pip install pynvml --upgrade --user
pip install tensorboard --upgrade --user
pip install PyYAML --upgrade --user
pip install accelerate==0.21.0 --upgrade --user
pip install peft==0.4.0 --upgrade --user
pip install typing_extensions==4.7.1 --upgrade --user
pip install transformers==4.31.0 --upgrade --user
pip install trl==0.4.7 --upgrade --user
pip install bitsandbytes==0.39.0 --upgrade --user
pip install jsonlines --upgrade --user
pip install fraction --upgrade --user
pip install protobuf==3.20.0 --upgrade --user
pip install git+https://github.com/PrithivirajDamodaran/Styleformer.git
pip install textattack

Note: Please upgrade the version of `transformers` in order to use the `mistralai/Mistral-7B-v0.1` model.

------------------------------------------------------

module load cuda/12.2

pip install pynvml --upgrade --user
pip install tensorboard --upgrade --user
pip install PyYAML --upgrade --user
pip install accelerate==0.21.0 --upgrade --user
pip install peft==0.6.0 --upgrade --user
pip install typing_extensions==4.7.1 --upgrade --user
pip install trl==0.4.7 --upgrade --user
pip install jsonlines --upgrade --user
pip install fraction --upgrade --user
pip install protobuf==3.20.0 --upgrade --user

pip3 install torch torchvision torchaudio --upgrade --user
pip install transformers==4.34.0 --upgrade --user
pip install bitsandbytes==0.41.1 --upgrade --user
pip install -U evaluate
pip install sentencepiece
pip install git+https://github.com/PrithivirajDamodaran/Styleformer.git --user
pip install googletrans==3.1.0a0 --upgrade --user
pip install nltk --upgrade --user
pip install sacremoses --upgrade --user
pip install sklearn_pandas --upgrade --user
pip install Levenshtein --upgrade --user
pip install scipy --upgrade --user
pip install scikit-learn --upgrade --user
pip install sentence_transformers --upgrade --user

------------------------------------------------------

git clone --branch support_peft https://github.com/SuperBruceJia/vllm.git
cd vllm
pip install -e . --user

------------------------------------------------------

git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes

CUDA_VERSION=122 make cuda12x
python setup.py install --user

------------------------------------------------------

cd /usr4/ec523/brucejia/.local/lib/python3.8/site-packages/transformers/generation

vim utils.py

# @torch.no_grad()
torch.set_grad_enabled(True)

------------------------------------------------------

Credits: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2697-L2734

cd /usr4/ec523/brucejia/.local/lib/python3.8/site-packages/transformers

vim trainer.py

def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    model.train()
    model.enable_adapters()
    torch.set_grad_enabled(True)
    
    inputs = self._prepare_inputs(inputs)
    
    if is_sagemaker_mp_enabled():
        loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        return loss_mb.reduce_mean().detach().to(self.args.device)
    
    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs)
    
    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    
    if self.do_grad_scaling:
        self.scaler.scale(loss).backward()
    elif self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        self.accelerator.backward(loss)
    torch.set_grad_enabled(False)
    
    return loss.detach() / self.args.gradient_accumulation_steps

------------------------------------------------------

https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L2578C13-L2584

---

# generation_mode == GenerationMode.SAMPLE:

https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1751C9-L1776C14

# Generates sequences of token ids for models with a language modeling head using **multinomial sampling**
# In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.

https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L2846-L2954

---

flake8 ./

---

pip install spacy --user
python -m spacy download en_core_web_sm

```

### 2. Selected Task
(1) **Grade School Math (GSM8K)**\
> [!IMPORTANT]
> Chain-of-Thought Prompting
> Few-shot In-Context Learning

**Description**: GSM8K consists of 8.5K high quality grade school math problems created by human problem writers. 
We segmented these into **7.5K training** problems and **1K test** problems. 
These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary 
calculations using basic arithmetic operations (+ - / *) to reach the final answer. 
A bright middle school student should be able to solve every problem.

**Format**: `.jsonl`

**Example**:
```shell
{"question": "Janet\u2019s ducks lay 16 eggs per day. 
She eats three for breakfast every morning and bakes muffins for her friends every day with four. 
She sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
How much in dollars does she make every day at the farmers' market?", 
"answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n#### 18"}
```

**Leaderboard**: [Arithmetic Reasoning on GSM8K](https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k)

**Load the database**:
```python
# Retrieve the path of training and testing databases
train_path = config.get("train_path")
test_path = config.get("test_path")

# Load datasets
context_feat = Features({"question": Value(dtype='string', id=None), "answer": Value(dtype='string', id=None)})
train_data = load_dataset('json', data_files=train_path, split='train', features=context_feat)
test_data = load_dataset('json', data_files=test_path, split='train', features=context_feat)
```

### 3. Selected Model
[LLaMA 2 from Meta AI](https://ai.meta.com/llama/) - `13.48GB`
```shell
model-00001-of-00002.safetensors: 100% - 9.98G/9.98G [01:15<00:00, 143MB/s]
model-00002-of-00002.safetensors: 100% - 3.50G/3.50G [00:19<00:00, 91.6MB/s]
```
**Note**: We will start with `Pre-trained LLaMA 2 7B model` (not the fine-tuned one for chat use cases).

### 4. Selected Fine-tuning Method
(1) [LoRA: Low-Rank Adaptation](https://arxiv.org/pdf/2106.09685.pdf)\
(2) [QLoRA: 4-bit Quantized Low Rank Adapters](https://arxiv.org/abs/2305.14314)

### Dependable Packages
All from 🤗 `Hugging Face`\
(1) transformers\
(2) peft\
(3) trl

For all the dependable packages, please check the `requirements.txt`.

## 💻 Download
### Sent this repo codes from GitHub to BU SCC Server 
```shell
# Account
SuperBruceJia
# Access Token
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

--------------------------------------------------------------------

# Export the TRANSFORMERS_CACHE, HF_HOME, and HF_DATASETS_CACHE paths
export TRANSFORMERS_CACHE=/projectnb/pnn/.cache
export HF_HOME=/projectnb/pnn/.cache
export HF_DATASETS_CACHE=/projectnb/pnn/.cache

# Clone this codes
git clone https://github.com/SuperBruceJia/IntuitLLMProject.git

# Show the GPUs available in the shared compute node queues
$ qgpus
$ qgpus -v

# Submit the job to SCC
$ qsub t1.qsub

# Submit another task before the before task
$ qsub t2.qsub
---
#$ -hold_jid t1
#$ -N t2
---

# Check the status of submitted jobs
$ watch -n 1 qstat -u brucejia

# Check the usage of GPU resources
$ watch -n 1 nvidia-smi

# Delete the job
$ qdel -j JOB_ID

# Monitor the progress
$ tail -f XXXXXX

# Check the disk quota
$ quota -s
$ du -s .[^.]* * | sort -n
$ pquota -u

# Check the memory usage
$ qstat -j JOB_ID

# Delete one user's files from folder /scratch
$ find /scratch -user brucejia -exec rm -fr {} \;

# Save all the terminal information to a txt file
$ python main.py not_find_runoob 2>&1 | tee -a terminal_result.txt

# pip install on scc
$ pip install --no-cache-dir --prefix=/projectnb/projectname/pythonlibs packagename
```
<details><summary>BU Available GPUs</summary>

![GPU.png](images/GPU.png)
![img.png](images/img.png)
![img_1.png](images/img_1.png)
![img_2.png](images/img_2.png)
![img_4.png](images/img_4.png)
</details>

```shell
# Change directory to the project folder
$ cd /project/pnn
```

<details><summary>BU Disk Quote</summary>

![img_5.png](images/img_5.png)
![img_6.png](images/img_6.png)
</details>

### Benchmark Database
#### Grade School Math (GSM8K)
```shell
$ git clone --recurse-submodules https://github.com/openai/grade-school-math.git
```
Path of the database: "./grade-school-math/grade_school_math/data/"

### Pre-trained Models
**On Google Colab**:
```shell
$ !huggingface-cli login --token "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"
```

**On the local machine**:\
(1) Access to `Pre-trained LLaMA 2` [here](https://ai.meta.com/llama/) by your 🤗 `Hugging Face` email\
(2) Generate a `Access Tokens` [here](https://huggingface.co/settings/tokens)\
(3) Log in your account through terminal
```shell
$ huggingface-cli login
```
**Note**: Later on you would receive an email like this: "This is to let you know your request to 
access model "meta-llama/Llama-2-70b-chat-hf" on huggingface.co has been accepted by the repo authors." 
Then you can use the `Pre-trained LLaMA 2` models, loaded by 🤗 `Hugging Face` packages, _e.g._, 
transformers and peft.

### Install Dependencies
```shell
$ pip install -r requirements.txt
```

## 🔧 Experiments
### 1. [Dataset] Dataset Pre-processing and Formatting
**Information of the original database**:
```shell
Training Set:
DatasetDict({
    train: Dataset({
        features: ['question', 'answer'],
        num_rows: 7473
    })
})

Testing Set:
DatasetDict({
    train: Dataset({
        features: ['question', 'answer'],
        num_rows: 1319
    })
})
```

**Prompt Template**:\
Please check the `gsm8k_prompt_g_format()` function in `utils.utils.py`.

**Data Formatting**:\
Please check the `dataset_loader()` function in `lib.data_manager.py`.

**Some Statistics**:\
**Tokenizer of the `GPT-2`**:\
**_Training set_**:

|          MODEL           | MAX | MIN | MEAN  | MEDIAN | STD  |
|:------------------------:|:---:|:---:|:-----:|:------:|:----:|
|        `Question`        | 213 | 10  | 55.2  |   51   | 20.7 |
|         `Answer`         | 342 | 23  | 95.3  |   86   | 42.1 |
|    `Question+Answer`     | 433 | 48  | 150.5 |  140   | 55.6 |
|    `Prompt+Question`     | 267 | 65  | 109.9 |  106   | 20.7 |
| `Prompt+Question+Answer` | 487 | 102 | 204.1 |  194   | 55.6 |

**_Testing set_**:

|          MODEL           | MAX | MIN | MEAN  | MEDIAN | STD  |
|:------------------------:|:---:|:---:|:-----:|:------:|:----:|
|        `Question`        | 182 | 21  | 56.8  |   53   | 21.5 |
|         `Answer`         | 302 | 23  | 97.8  |   91   | 42.0 |
|    `Question+Answer`     | 400 | 57  | 154.6 |  146   | 55.8 |
|    `Prompt+Question`     | 237 | 76  | 111.5 |  107   | 21.5 |
| `Prompt+Question+Answer` | 454 | 110 | 208.2 |  200   | 55.9 |

**Tokenizer of the `LLaMA 2 - 7B`**:

**_Training set_**:

|          MODEL           | MAX | MIN | MEAN  | MEDIAN | STD  |
|:------------------------:|:---:|:---:|:-----:|:------:|:----:|
|        `Question`        | 239 | 12  | 65.8  |   61   | 24.3 |
|         `Answer`         | 475 | 29  | 129.6 |  118   | 57.0 |
|    `Question+Answer`     | 570 | 66  | 195.4 |  182   | 72.7 |
|    `Prompt+Question`     | 298 | 71  | 124.8 |  120   | 24.3 |
| `Prompt+Question+Answer` | 628 | 124 | 253.4 |  240   | 72.7 |

**_Testing set_**:

|          MODEL           | MAX | MIN | MEAN  | MEDIAN | STD  |
|:------------------------:|:---:|:---:|:-----:|:------:|:----:|
|        `Question`        | 190 | 23  | 67.4  |   63   | 25.0 |
|         `Answer`         | 428 | 29  | 131.8 |  122   | 56.7 |
|    `Question+Answer`     | 569 | 70  | 199.3 |  190   | 72.6 |
|    `Prompt+Question`     | 249 | 82  | 126.4 |  122   | 25.0 |
| `Prompt+Question+Answer` | 627 | 128 | 257.3 |  248   | 72.6 |

Please check `data_analysis.gsm8k_token_num.py` for details about codes.

**Information after the pre-processing**:
```shell
Training set:
DatasetDict({
    train: Dataset({
        features: ['question', 'answer', 'text'],
        num_rows: 7473
    })
})

Testing Set:
DatasetDict({
    train: Dataset({
        features: ['question', 'answer', 'text'],
        num_rows: 1319
    })
})
```

### 2. [Dataset] Data Augmentation by Hierarchal Prompt Perturbations
#### (1) `Character Editing` - 5-level Severity -> 30 sentences
    [1] Character Replacement
    [2] Character Deletion
    [3] Character Insertion
    [4] Character Swap
    [5] Keyboard Typos
    [6] Optical Character Recognition (OCR)
"_simulating real-world accidental typos or spelling variants during typing_"\
Please check `data_augmentation.character.py` for more details.

#### (2) `Word Manipulation` - 5-level Severity -> 30 sentences
    [1] Synonym Replacement
    [2] Word Insertion
    [3] Word Swap
    [4] Word Deletion
    [5] Insert Punctuation
    [6] Word Split
"_simulating different writing habits that people may replace, delete, 
or add words to express the same meaning_"\
Please check `data_augmentation.word.py` for more details.

**Note**:\
1> As for the `Synonym Replacement`, `Word Insertion`, we can also insert word randomly by 
(contextual) word embeddings / TF-IDF similarity. 

#### (3) `Sentence Paraphrasing` -> 6 sentences
    [1] Back Translation based on 🤗 Hugging Face MarianMTModel
    [2] Back Translation based on Google Translator
    [3] Style Transformation to Formal
    [4] Style Transformation to Casual
    [5] Style Transformation to Passive
    [6] Style Transformation to Active
"_simulating more language semantics, due to the differences 
of speaking/writing styles, or translation errors_"\
Please check `data_augmentation.sentence.py` for more details.

#### (4) `Adversarial Augmentation`
    Reference: https://github.com/QData/TextAttack    
    Transformation -> Search with constraints -> Attack based on the goal function
"_simulating intentional modifications of input data that aim to 
increase model generalization and robustness downstream_"

### 3. [`Problem 1` - Dataset] Semantics-preserving Evaluation
    [1] Sentence-BERT/Transformers
    [2] BERT-Score
    [3] Universal Sentence Encoder
    [4] LLM prompt, e.g., ChatGPT

Please check `semantics_evaluation.semantics_similarity.py` for more details.

### 4. [`Problem 2` - Dataset] Consistency Comparison (with the answer of base prompt)
TODO

### 5. [Dataset] Database Construction (train/dev/test)
TODO

### 6. [`Problem 3` - Model Input] Prompt Template (for Discriminator and Generator)

For Discriminator $D$:

```python
answer = re.sub('[\n\t]', ' ', answer)
answer = answer.replace('####', 'The answer is:')
result_format = " \n\n### Is it the right answer to the question?"
prompt = (f'{"Below is an instruction that describes a task, and the response is as follows."}'
          + "\n\n### Instruction:\n" + question
          + "\n\n### Response: " + answer + result_format)
```
Please check the `gsm8k_prompt_d_format()` function in `utils.utils.py`.

For Generator $G$:

```python
question = re.sub('[\n\t]', ' ', question)
question = question.replace('  ', ' ')
prompt = (f'{"Below is an instruction that describes a task. "}'
          f'{"Write a response that appropriately completes the request."}'
          + "\n\n### Instruction:\n" + question
          + "\n\n### Response: Let's think step by step.")

if inference is False:
    answer = re.sub('[\n\t]', ' ', answer)
    answer = answer.replace('####', 'The answer is:')
    prompt += answer
else:
    pass
```

### 7. [Model] Adversarial Learning (Generator part and updating Generator)

> [!IMPORTANT]
> The following two items are important to fine-tune the LLMs, _i.e._, updating the Generator $G$
> 
> [1] Use the **Structured Data** to train the model, _e.g._, "### Instruction. ### Response"
> 
> [2] Use a **better Optimizer**. I used `paged_adamw_8bit`

Transform the **Binary Classification** task into a **Text Generation** task

**Reason**: It is non-trivial to back-prop the gradients to the Generator $G$ from the Discriminator $D$.  

**Our approach**: As long as $y_{gen}$ ~ $y$, $D(y_{gen}) \approx D(y) = 1$.

We will use the *mean score value* of the Discriminator $D$ as a condition 
to stop the training process of the Generator $G$.

### 8. [Model] Adversarial Learning (Discriminator part and updating Discriminator)
**Binary Classification Task**

**Real sample**: Input: ($q$, $y$), Output: 1\
**Fake sample**: Input: ($q$, $y_{gen}$), Output:  0

### 9. [Evaluation Metrics] Performance Evaluation (Accuracy and Consistency)
(1) **Accuracy** between the final predicted numerical number and gold label

(2) **Consistency** among the same question perturbations

TODO

## 🏋️‍️ Online Inference
### Download Checkpoint
Please put the checkpoint into `models` folder.

### Run
```shell
$ python inference.py --question "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
```

## License
This repository is licensed under the [MIT License](LICENSE).\
Please follow the model licenses to use the corresponding model weights: 
[LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) / 
[LLaMA-2](https://ai.meta.com/llama/license/)

## Acknowledgement
We appreciate Dr. Jiaxin Zhang and Dr. Weichao Zhou for helpful discussions.
