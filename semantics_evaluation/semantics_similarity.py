# coding=utf-8

import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer, util
from bert_score import score

# Load the Universal Sentence Encoder model
sen_tf_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
use_model = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_embed = hub.load(use_model)


class SemanticsSimilarity:
    def __init__(self, base_sen, aug_sen):
        self.base_sen = base_sen
        self.aug_sen = aug_sen

    def sentence_transformers(self):
        """
        We use paraphrases from pretrained sentence-transformers (Sentence-BERT) to
        evaluate the semantic similarity between the original and perturbed sentences.

        Credits:
        https://github.com/Jason-Qiu/MM_Robustness/blob/main/text_perturbation/perturb_COCO_TP.py#L392C30-L392C30
        """
        # Input base sentence and the augmented sentence
        base_sen = self.base_sen[:]
        aug_sen = self.aug_sen[:]

        # Get the embedding of the sentences
        embed_base = sen_tf_model.encode(base_sen)
        embed_aug = sen_tf_model.encode(aug_sen)

        # Calculate the similarity score
        sim_score = float(util.cos_sim(embed_aug, embed_base))

        return sim_score

    def bert_score(self):
        # Input base sentence and the augmented sentence
        base_sen = self.base_sen[:]
        aug_sen = self.aug_sen[:]

        # Calculate the similarity score
        # Some contextual embedding models, like RoBERTa, often produce BERTScores in a very narrow range
        # (e.g., the range is roughly between 0.92 and 1).
        # Although this artifact does not affect the ranking ability of BERTScore, it affects the readability.
        # Therefore, we propose to apply "baseline rescaling" to adjust the output scores.
        P, R, F1 = score([aug_sen], [base_sen], lang='en', rescale_with_baseline=True)

        return P, R, F1

    def universal_sentence_encoder(self):
        # Input base sentence and the augmented sentence
        base_sen = self.base_sen[:]
        aug_sen = self.aug_sen[:]

        # Encode the sentences into embeddings
        embeddings = use_embed([base_sen, aug_sen])

        # Calculate the cosine similarity between the two embeddings
        similarity = tf.tensordot(embeddings[0], embeddings[1], axes=1)

        return similarity.numpy()


if __name__ == "__main__":
    # Sentence and its augmentation
    base_sen = "I like USA, Hong Kong SAR@ and China!!!!!!!!!"
    aug_sen = "I don't like the United States of America, Hong Kong, and CHINA!"

    sen_sim_tool = SemanticsSimilarity(base_sen=base_sen, aug_sen=aug_sen)

    sentence_bert_score = sen_sim_tool.sentence_transformers()
    bert_score_p, bert_score_r, bert_score_f1 = sen_sim_tool.bert_score()
    use_score = sen_sim_tool.universal_sentence_encoder()

    print('Sentence BERT/Transformers: ', sentence_bert_score)
    print('BERT Score: ', bert_score_p, bert_score_r, bert_score_f1)
    print('Universal Sentence Encoder: ', use_score)
