# coding=utf-8

import os
import jsonlines

from data_processing.paragraph_split import paragraph_splitter
from data_augmentation.character import CharacterPerturb
from data_augmentation.word import WordPerturb
from data_augmentation.sentence import SentencePerturb
from semantics_evaluation.semantics_similarity import SemanticsSimilarity
from utils.utils import load_config

os.chdir("../")
config = load_config()
train_path = config.get("train_path")


perturbation = []
data_path = train_path
with open(data_path, "r+", encoding="utf8") as f:
    for idx, item in enumerate(jsonlines.Reader(f)):
        paragraph = item["question"]
        sentences = paragraph_splitter(paragraph=paragraph)

        sentence = sentences[0]
        levels = [0.05, 0.10, 0.15, 0.20, 0.25]
        for lev in levels:
            # Character-level Perturbation
            character_tool = CharacterPerturb(sentence=sentence, level=lev)

            char_replace = character_tool.character_replacement()
            char_delete = character_tool.character_deletion()
            char_insert = character_tool.character_insertion()
            char_swap = character_tool.character_swap()
            char_keyboard = character_tool.keyboard_typos()
            char_ocr = character_tool.optical_character_recognition()

            perturbation.append(char_replace)
            perturbation.append(char_delete)
            perturbation.append(char_insert)
            perturbation.append(char_swap)
            perturbation.append(char_keyboard)
            perturbation.append(char_ocr)

            # Word-level Perturbation
            word_tool = WordPerturb(sentence=sentence, level=lev)

            word_synonym = word_tool.synonym_replacement()
            word_insert = word_tool.word_insertion()
            word_swap = word_tool.word_swap()
            word_delete = word_tool.word_deletion()
            word_punctuation = word_tool.insert_punctuation()
            word_split = word_tool.word_split()

            perturbation.append(word_synonym)
            perturbation.append(word_insert)
            perturbation.append(word_swap)
            perturbation.append(word_delete)
            perturbation.append(word_punctuation)
            perturbation.append(word_split)

        # Sentence-level Perturbation
        sentence_tool = SentencePerturb(sentence=sentence)
        back_trans_hf = sentence_tool.back_translation_hugging_face()
        back_trans_google = sentence_tool.back_translation_google()
        sen_formal = sentence_tool.formal()
        sen_casual = sentence_tool.casual()
        sen_passive = sentence_tool.passive()
        sen_active = sentence_tool.active()

        perturbation.append(back_trans_hf)
        perturbation.append(back_trans_google)
        perturbation.append(sen_formal)
        perturbation.append(sen_casual)
        perturbation.append(sen_passive)
        perturbation.append(sen_active)

        print(perturbation)

        for sen in perturbation:
            sen_sim_tool = SemanticsSimilarity(base_sen=sentence, aug_sen=sen)

            sentence_bert_score = sen_sim_tool.sentence_transformers()
            bert_score_p, bert_score_r, bert_score_f1 = sen_sim_tool.bert_score()
            use_score = sen_sim_tool.universal_sentence_encoder()

            print('Original Sentence:\n', sentence)
            print('Perturbed Sentence:\n', sen)
            print('Sentence BERT/Transformers: ', sentence_bert_score)
            print('BERT Score: Precision: ', bert_score_p, 'Recall: ', bert_score_r, 'F1-socre: ', bert_score_f1)
            print('Universal Sentence Encoder: ', use_score, '\n\n')

        break
