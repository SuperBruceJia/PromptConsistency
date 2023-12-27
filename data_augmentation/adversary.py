# coding=utf-8

# import transformers
# import textattack
# from textattack import Attack, AttackArgs
# from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
# from textattack.constraints.semantics import WordEmbeddingDistance
# from textattack.transformations import WordSwapEmbedding, BackTranslation
# from textattack.search_methods import GreedyWordSwapWIR, GreedySearch
from textattack.constraints.semantics import BERTScore
# import transformations, contraints, and the Augmenter
from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.transformations import WordSwapQWERTY
from textattack.transformations import CompositeTransformation
# from textattack.constraints.pre_transformation import RepeatModification, StopwordModification, MaxModificationRate
# from textattack.constraints.pre_transformation import StopwordModification
from textattack.augmentation import Augmenter

# Set up transformation using CompositeTransformation()
transformation = CompositeTransformation([
    WordSwapRandomCharacterDeletion(),
    WordSwapQWERTY(),
])

# Set up constraints
constraints = [
    # Constraint using similarity between sentence encodings of x and x_adv
    # where the text embeddings are created using the Universal Sentence Encoder.
    # sentence_encoders.universal_sentence_encoder.universal_sentence_encoder.UniversalSentenceEncoder(threshold=0.9),

    # Constraint using cosine similarity between sentence encodings of x and x_adv.
    # sentence_encoders.SentenceEncoder(threshold=0.9),

    # BERT Score measures token similarity between two text using contextual embedding.
    # To decide which two tokens to compare, it greedily chooses the most similar token from one text
    # and matches it to a token in the second text.
    # paraphrase-mpnet-base-v2
    BERTScore(min_bert_score=0.9, model_name='bert-base-uncased'),

    # sBERT for Sentence Similarity
    # Constraint using similarity between sentence encodings of x and x_adv where the text embeddings
    # are created using BERT, trained on NLI data, and fine-tuned on the STS benchmark dataset.
    # sentence_encoders.bert.bert.BERT(threshold=0.9, model_name='bert-base-nli-stsb-mean-tokens'),

    # Constraint using similarity between sentence encodings of x and x_adv
    # where the text embeddings are created using the Multilingual Universal Sentence Encoder.
    # sentence_encoders.universal_sentence_encoder.multilingual_universal_sentence_encoder.MultilingualUniversalSentenceEncoder(
    #     threshold=0.9)
]

# Create augmenter with specified parameters
augmenter = Augmenter(transformation=transformation,
                      constraints=constraints,
                      pct_words_to_swap=0.5,
                      transformations_per_example=100)

sentence = ("Transformations perfectly preserve syntax or semantics, so additional constraints can "
            "increase the probability that these qualities are preserved from the source to adversarial example.")

# Augment!
augmenter.augment(sentence)

# Load model, tokenizer, and model_wrapper
# bert-base-uncased and bert-base-cased
# distilbert-base-uncased and distilbert-base-cased
# albert-base-v2
# roberta-base
# xlnet-base-cased
# https://textattack.readthedocs.io/en/latest/_modules/textattack/model_args.html?highlight=textattack%2Fbert-base-uncased-imdb#:~:text=ARGS_SPLIT_TOKEN%2C%20load_module_from_file-,HUGGINGFACE_MODELS,-%3D%20%7B%0A%20%20%20%20%23
# model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
# tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
# model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# Construct our four components for `Attack`
# goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
# attack_args = AttackArgs(num_examples=10)

# [1] bert (textattack.constraints.semantics.sentence_encoders.BERT)
# [2] bert-score (textattack.constraints.semantics.BERTScore)
# [7] embedding (textattack.constraints.semantics.WordEmbeddingDistance)
# [10] infer-sent (textattack.constraints.semantics.sentence_encoders.InferSent)
# [16] muse (textattack.constraints.semantics.sentence_encoders.MultilingualUniversalSentenceEncoder)
# [20] thought-vector (textattack.constraints.semantics.sentence_encoders.ThoughtVector)
# [21] use (textattack.constraints.semantics.sentence_encoders.UniversalSentenceEncoder)

# [3] bleu (textattack.constraints.overlap.BLEU)
# [4] chrf (textattack.constraints.overlap.chrF)
# [5] cola (textattack.constraints.grammaticality.COLA)
# [6] edit-distance (textattack.constraints.overlap.LevenshteinEditDistance)
# [8] goog-lm (textattack.constraints.grammaticality.language_models.GoogleLanguageModel)
# [9] gpt2 (textattack.constraints.grammaticality.language_models.GPT2)
# [11] lang-tool (textattack.constraints.grammaticality.LanguageTool)
# [12] learning-to-write (textattack.constraints.grammaticality.language_models.LearningToWriteLanguageModel)
# [13] max-word-index (textattack.constraints.pre_transformation.MaxWordIndexModification)
# [14] max-words-perturbed (textattack.constraints.overlap.MaxWordsPerturbed)
# [15] meteor (textattack.constraints.overlap.METEOR)
# [17] part-of-speech (textattack.constraints.grammaticality.PartOfSpeech)
# [18] repeat (textattack.constraints.pre_transformation.RepeatModification)
# [19] stopword (textattack.constraints.pre_transformation.StopwordModification)

# constraints = [
#     BERTScore(min_bert_score=0.9)
#     # RepeatModification(),
#     # StopwordModification(),
#     # WordEmbeddingDistance(min_cos_sim=0.9)
# ]

# transformation = WordSwapEmbedding(max_candidates=100)
# transformation = BackTranslation()
# transformation = WordSwapEmbedding(max_candidates=50)


# [1] beam-search (textattack.search_methods.BeamSearch)
# [2] ga-word (textattack.search_methods.GeneticAlgorithm)
# [3] greedy (textattack.search_methods.GreedySearch)
# [4] greedy-word-wir (textattack.search_methods.GreedyWordSwapWIR)
# [5] pso (textattack.search_methods.ParticleSwarmOptimization)
# search_method = GreedySearch()
# search_method = GreedyWordSwapWIR(wir_method="delete")

# Construct the actual attack

# The goal function determines if the attack is successful or not.
# One common goal function is untargeted classification,
# where the attack tries to perturb an input to change its classification.

# The search method explores the space of potential transformations and tries to locate a successful perturbation.
# Greedy search, beam search, and brute-force search are all examples of search methods.

# A transformation takes a text input and transforms it, for example replacing words or phrases with similar ones,
# while trying not to change the meaning. Paraphrase and synonym substitution are two broad classes of transformations.

# constraints determine whether or not a given transformation is valid.
# Transformations don’t perfectly preserve syntax or semantics, so additional constraints can increase the probability
# that these qualities are preserved from the source to adversarial example.
# There are many types of constraints: overlap constraints that measure edit distance, syntactical constraints
# check part-of-speech and grammar errors, and semantic constraints like language models and sentence encoders.
# attack = Attack(goal_function, constraints, transformation, search_method)

# input_text = "I really enjoyed the new movie that came out last month."
# sentence = ("Transformations perfectly preserve syntax or semantics, so additional constraints
# can increase the probability "
#             "that these qualities are preserved from the source to adversarial example.")
# label = 1  # Positive
# attack_result = attack.attack(sentence, label)
#
# print(attack_result)
