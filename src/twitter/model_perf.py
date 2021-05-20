
import pandas as pd
from sklearn.metrics import confusion_matrix

import requests
import os
import json
from datetime import date, datetime
import numpy as np

import flair  # sentiment analysis
import spacy
from spacymoji import Emoji
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacytextblob.spacytextblob import SpacyTextBlob  # sentiment analysis
import contractions

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')  # load outside


def get_score(texts):  # texts: list of strings
    expanded_texts = []
    for t in texts:
        print(t)
        expanded_texts.append(contractions.fix(t))

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("emoji", first=True)
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # Commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    docs = list(nlp.pipe(expanded_texts))

    lst_cleaned = []  # list of list of tweet text tokens
    for doc in docs:
        # construct a list of valid wanted Tokens from the raw_sentence
        token_lst = []
        for token in doc:
            if token._.is_emoji:  # emoji and special graphic chars e.g. ♂️
                continue
            if token.is_space:  # newline chars
                continue
            if token.like_url:
                continue
            if token.is_punct:
                continue
            if token.text[0] == '@':  # name tags
                continue
            if token.is_currency:  # '$'
                continue
            if token.pos_ == 'NUM' or token.is_digit:  # number words or numbers
                continue
            token_lst.append(token)
        clean_sentence_lst = []  # list of strings
        for token in token_lst:
            clean_sentence_lst.append(token.lemma_)
        lst_cleaned.append(clean_sentence_lst)
    
    print("flair making sentences", datetime.now())
    lst_sentences = []
    for lst in lst_cleaned:
        # print("making sentence")
        lst_sentences.append(flair.data.Sentence(lst))
    
    flair_sentiment.predict(lst_sentences, verbose=True, mini_batch_size=128)
    lst_flair = []
    for sentence in lst_sentences:
        score = sentence.labels[0].score
        if sentence.labels[0].value == "NEGATIVE":
            score = score - 2 * score
        lst_flair.append(score)
    
    nlp.add_pipe('spacytextblob')
    lst_concat = []
    print("spacy for loop concat", datetime.now())
    for text in lst_cleaned:
        clean_str = ""
        for s in text:
            clean_str += s + " "
        lst_concat.append(clean_str)
    docs = list(nlp.pipe(lst_concat))
    lst_spacy = []  # list of floats for sentiment scores
    for doc in docs:
        lst_spacy.append(doc._.polarity)

    return (lst_flair, lst_spacy, lst_concat, texts)

def get_idx_arr(flair_scores, threshold):
    raw_scores = np.array(flair_scores)
    idx_arr = np.nonzero(raw_scores < threshold)
    return idx_arr[0]

# labelled dataset from kaggle: https://www.kaggle.com/kazanova/sentiment140
df = pd.read_csv('./training.1600000.processed.noemoticon.csv')


lst_texts = df.iloc[:, 5]  # tweet texts
lst_labels = df.iloc[:, 0]  # tweet sentiment  0:neg, 2:neutral, 4:positive

scores = get_score(lst_texts)
flair_scores = scores[0]
spacy_scores = scores[1]
flair_negs = get_idx_arr(flair_scores, -0.9)
spacy_negs = get_idx_arr(spacy_scores, -0.5)

len_dataset = len(lst_labels)

flair_classified = []
for i in range(len_dataset):
    if i in flair_negs:
        flair_classified.append(0)  # negative
    else:
        flair_classified.append(1)

spacy_classified = []
for i in range(len_dataset):
    if i in spacy_negs:
        spacy_classified.append(0)  # negative
    else:
        spacy_classified.append(1)

# process lst_labels to be negatives (0) or otherwise (1)
for i in range(len_dataset):
    if lst_labels[i] != 0:
        lst_labels[i] = 1

tn, fp, fn, tp = confusion_matrix(lst_labels, flair_classified)
print(tn, fp, fn, tp)



