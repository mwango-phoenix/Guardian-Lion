import requests
import os
import json
from datetime import date
import numpy as np

import flair  # sentiment analysis
import spacy
from spacymoji import Emoji
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacytextblob.spacytextblob import SpacyTextBlob  # sentiment analysis
import contractions


# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'


def auth():
    return os.environ.get("BEARER_TOKEN")


def create_url(query, tweet_fields, next_token):
    url = "https://api.twitter.com/2/tweets/search/recent?query={}&max_results=10&{}".format(query, tweet_fields)
    # max_results can be adjusted 10-100
    if next_token != '':
        url = url + '&next_token={}'.format(next_token)
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print("status code", response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )

    return response.json()


def get_file_name(query, token=''):
    return '../../data/{}_{}_{}.json'.format(date.today().strftime("%Y_%m_%d"), query, token)


# in batches, get Tweets (json_response) and feed into clean_up & output scores from 2 models
def get_data(bearer_token, query, tweet_fields, max_items):  # -> Tuple[List[float]]
    next_token = ''
    total_items = 0
    data = {'data' : []}
    filename = get_file_name(query)
    flair_scores = []
    spacy_scores = []
    cleaned_texts = []
    while total_items < max_items:
        url = create_url(query, tweet_fields, next_token)
        headers = create_headers(bearer_token)
        json_response = connect_to_endpoint(url, headers)
        lst_texts = []
        for i in range(len(json_response['data'])):
            # print(json_response['data'][i])
            if 'referenced_tweets' in json_response['data'][i]:
                original_tweet_id = json_response['data'][i]['referenced_tweets'][0]['id']
                # find the original tweet by id
                url = "https://api.twitter.com/2/tweets/{}?tweet.fields=text,author_id".format(original_tweet_id)
                original_tweet_res = connect_to_endpoint(url, headers)
                json_response['data'][i]['referenced_tweets'][0]['user_id'] = original_tweet_res['data']['author_id']
                json_response['data'][i]['referenced_tweets'][0]['text'] = original_tweet_res['data']['text']

                # scores = get_score(json_response['data'][i]['referenced_tweets'][0]['text'])
                # flair_scores.append(scores[0])
                # spacy_scores.append(scores[1])
                # cleaned_texts.append(scores[2])
                lst_texts.append(json_response['data'][i]['referenced_tweets'][0]['text'])
            else:
                # scores = get_score(json_response['data'][i]['text'])
                # flair_scores.append(scores[0])
                # spacy_scores.append(scores[1])
                # cleaned_texts.append(scores[2])
                lst_texts.append(json_response['data'][i]['text'])
        scores = get_score(lst_texts)  # lst of 10 tuples: (model1_score, model2_score, cleaned_str)

        total_items = total_items + len(json_response['data'])
        data['data'] = data['data'] + json_response['data']
        # print(json_response)
        print(data)

        # print(json.dumps(json_response, indent=4, sort_keys=True))
        # with open(filename, "w") as f:
        #     f.write(json.dumps(data, indent=4, sort_keys=True))

        # filename_token = get_file_name(query, next_token)
        # with open(filename_token, "w") as f:
        #     f.write(json.dumps(json_response, indent=4, sort_keys=True))

        next_token = json_response['meta']['next_token']
    return scores


def get_score(texts):  # texts: list of strings
    expanded_texts = []
    for t in texts:
        # expand contraction
        expanded_texts.append(contractions.fix(t))

    # tokenization: raw_sentence string -> List[Token]
    # load default tokenizer
    nlp = spacy.load("en_core_web_sm")
    # add the emoji detection component spacymoji
    nlp.add_pipe("emoji", first=True)
    # Modify tokenizer infix patterns so that we don't split on hyphens between letters
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

    # tokenize
    # print(expanded_texts)
    docs = list(nlp.pipe(expanded_texts))
    # print(docs[0])
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

        # lemmatization
        clean_sentence_lst = []  # list of strings
        for token in token_lst:
            clean_sentence_lst.append(token.lemma_)
        lst_cleaned.append(clean_sentence_lst)
    
    # print(lst_cleaned)

    # using Flair for sentiment analysis
    lst_sentences = []
    for lst in lst_cleaned:
        lst_sentences.append(flair.data.Sentence(lst))
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    flair_sentiment.predict(lst_sentences)
    lst_flair = []
    for sentence in lst_sentences:
        lst_flair.append(sentence.labels)
    # print("flair_sentiment", total_sentiment)


    # using SpaCy's Textblob for sentiment analysis
    # https://spacytextblob.netlify.app/docs/example, can be for multiple texts
    nlp.add_pipe('spacytextblob')
    lst_concat = []
    for text in lst_cleaned:
        clean_str = ""
        for s in text:
            clean_str += s + " "
        lst_concat.append(clean_str)
    docs = list(nlp.pipe(lst_concat))
    lst_spacy = []  # list of floats for sentiment scores
    for doc in docs:
        lst_spacy.append(doc._.polarity)
    # print("spacy_sentiment", doc._.polarity)

    return (lst_flair, lst_spacy, lst_concat)



def main():
    bearer_token = auth()

    # Tweet fields are adjustable. see https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id (tweet id of the original tweet), created_at, entities, geo, id,
    # in_reply_to_user_id (user id of original tweet), 
    # lang, non_public_metrics, organic_metrics,
    # possibly_sensitive (of the url contained in tweet), 
    # promoted_metrics (inc. like/reply counts, but requires user context auth),
    # public_metrics, referenced_tweets,
    # source, text, and withheld
    tweet_fields = "tweet.fields=text,author_id,lang,referenced_tweets,in_reply_to_user_id,conversation_id,possibly_sensitive,created_at,public_metrics"  

    # search term
    query = 'lang:en "china virus"'

    max_items = 10

    tup_scores = get_data(bearer_token, query, tweet_fields, max_items)
    print(tup_scores[0])
    print(tup_scores[1])
    print(tup_scores[2])


if __name__ == "__main__":
    main()
