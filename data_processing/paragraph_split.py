# coding=utf-8

import nltk

nltk.download('punkt', download_dir="./save_folder/nltk")


def paragraph_splitter(paragraph):
    sentences = nltk.sent_tokenize(paragraph)

    return sentences


# if __name__ == "__main__":
#     # A paragraph
#     paragraph = ("Janet’s ducks lay 16 eggs per day. "
#                  "She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
#                  "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
#                  "How much in dollars does she make every day at the farmers' market? The answer is 56")
#
#     sentences = sentences_splitter(paragraph)
#     print(sentences)
