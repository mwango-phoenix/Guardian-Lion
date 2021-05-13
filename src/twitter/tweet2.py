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

# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'


def auth():
    return os.environ.get("BEARER_TOKEN")


def create_url(query, tweet_fields, next_token):
    # max_results can be adjusted 10-100
    # user.field returns username by default
    url = "https://api.twitter.com/2/tweets/search/recent?query={}&max_results=100&{}&expansions=author_id&user.fields=public_metrics".format(query, tweet_fields)
    if next_token != '':
        url = url + '&next_token={}'.format(next_token)
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    # print("status code", response.status_code)
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
    # print("entering get_data", datetime.now())
    next_token = ''
    total_items = 0
    data = {'data' : []}
    filename = get_file_name(query)
    # lst_scores = []
    flair_scores = []
    spacy_scores = []
    cleaned_texts = []
    raw_texts = []
    while total_items < max_items:
        # print("entering while loop/api", datetime.now())
        url = create_url(query, tweet_fields, next_token)
        headers = create_headers(bearer_token)
        json_response = connect_to_endpoint(url, headers)
        lst_texts = []
        lst_i = []
        # print(len(json_response['data']), len(json_response['includes']['users']))
        for i in range(len(json_response['data'])):
            # if 'referenced_tweets' in json_response['data'][i]:
            #     original_tweet_id = json_response['data'][i]['referenced_tweets'][0]['id']
            #     # find the original tweet by id
            #     url = "https://api.twitter.com/2/tweets/{}?tweet.fields=text,author_id".format(original_tweet_id)
            #     try:
            #         original_tweet_res = connect_to_endpoint(url, headers)
            #         if 'data' in original_tweet_res:
            #             json_response['data'][i]['referenced_tweets'][0]['user_id'] = original_tweet_res['data']['author_id']
            #             json_response['data'][i]['referenced_tweets'][0]['text'] = original_tweet_res['data']['text']
            #             lst_texts.append(json_response['data'][i]['referenced_tweets'][0]['text'])
            #             if total_items + len(lst_texts) >= max_items:
            #                 break
            #         else:  # case: unable to view the original tweet
            #             continue
            #     except:
            #         continue
            # else:
            #     lst_texts.append(json_response['data'][i]['text'])
            #     if total_items + len(lst_texts) >= max_items:
            #         break

            # approach: don't retrieve original tweet text, in order to speed up
            # clean up retweet header
            if 'referenced_tweets' in json_response['data'][i] and json_response['data'][i]['referenced_tweets'][0]["type"] == "retweeted":
                lst_i.append(json_response['data'][i])
            else:
                lst_texts.append(json_response['data'][i]['text'])
                # number of users in json_response['includes']['users'] might be greater than number of tweets
                # append info about author to json_response
                for item in json_response['includes']['users']:
                    if item['id'] == json_response['data'][i]['author_id']:
                        json_response['data'][i]['username'] = item['username']
                        json_response['data'][i]['name'] = item['name']
                        json_response['data'][i]['tweet_count'] = item['public_metrics']['tweet_count']
                        json_response['data'][i]['followers_count'] = item['public_metrics']['followers_count']
                        break
            if total_items + len(lst_texts) >= max_items:
                break

        # print(len(lst_texts))
        # lst_scores.append(get_score(lst_texts))  # lst of 100 tuples: (model1_score, model2_score, cleaned_str)
        batch_scores = get_score(lst_texts)
        flair_scores.extend(batch_scores[0])
        spacy_scores.extend(batch_scores[1])
        cleaned_texts.extend(batch_scores[2])
        raw_texts.extend(batch_scores[3])

        # print("*********done", datetime.now())

        total_items += len(lst_texts)
        print("total_items", total_items)
        print("before", len(json_response['data']))
        for item in lst_i:
            json_response['data'].remove(item)
        print("after", len(json_response['data']))
            
        data['data'] = data['data'] + json_response['data']  # this json_response includes tweets with original unable to retrieve
        # print(json_response)
        # print(data)

        # print(json.dumps(json_response, indent=4, sort_keys=True))
        # with open(filename, "w") as f:
        #     f.write(json.dumps(data, indent=4, sort_keys=True))

        # filename_token = get_file_name(query, next_token)
        # with open(filename_token, "w") as f:
        #     f.write(json.dumps(json_response, indent=4, sort_keys=True))

        next_token = json_response['meta']['next_token']

    idx_arr = get_idx_arr(flair_scores)  # classify -> [0,1,1,0,...]
    neg_rate = len(idx_arr)/len(flair_scores)
    print("neg_rate", neg_rate)

    neg_users = find_neg_users(idx_arr, data['data'])
    print("neg_users:", neg_users)  # sorted
    return (flair_scores, spacy_scores, cleaned_texts, raw_texts)

def find_neg_users(idx_arr, data):
    neg_users = {}
    for idx in idx_arr:
        tweet = data[idx]
        user = tweet['author_id']
        if user not in neg_users:
            # [username, # of total tweets, # of followers, # of retweets, #of likes, # of neg tweets found, texts of neg tweets found]
            neg_users[user] = [tweet['username'], tweet['tweet_count'], tweet['followers_count'], 0, 0, 0, []]
        # if tweet['public_metrics']['retweet_count'] > 10:
        #     print(tweet['username'])
        neg_users[user][3] += tweet['public_metrics']['retweet_count']
        neg_users[user][4] += tweet['public_metrics']['like_count']
        neg_users[user][5] += 1
        neg_users[user][6].append(tweet['text'])

        # sort by number of likes and number of followers
        neg_users = dict(sorted(neg_users.items(), key=lambda item: (item[1][4], item[1][2]), reverse=True))

    return neg_users

# classify
def get_idx_arr(flair_scores):
    raw_scores = np.array(flair_scores)
    idx_arr = np.nonzero(raw_scores < -0.9)
    return idx_arr[0]


def get_score(texts):  # texts: list of strings
    # print("entering get_score", datetime.now())
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

    # print("tokenize", datetime.now())

    # tokenize
    # print(expanded_texts)
    docs = list(nlp.pipe(expanded_texts))
    
    # print("filtering nested for loop", datetime.now())
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
    print("flair making sentences", datetime.now())
    lst_sentences = []
    for lst in lst_cleaned:
        # print("making sentence")
        lst_sentences.append(flair.data.Sentence(lst))
    
    # print("flair predicting", datetime.now())
    # lst_sentences.extend(lst_sentences)  # => linear
    flair_sentiment.predict(lst_sentences, verbose=True, mini_batch_size=128)
    # print("finished predicting")
    
    # print("flair labels appending to lst_flair", datetime.now())
    lst_flair = []
    for sentence in lst_sentences:
        score = sentence.labels[0].score
        if sentence.labels[0].value == "NEGATIVE":
            score = score - 2 * score
        lst_flair.append(score)
    # print(lst_flair)


    # using SpaCy's Textblob for sentiment analysis; which utilizes nltk
    # https://spacytextblob.netlify.app/docs/example, can be for multiple texts
    
    nlp.add_pipe('spacytextblob')
    lst_concat = []
    print("spacy for loop concat", datetime.now())
    for text in lst_cleaned:
        clean_str = ""
        for s in text:
            clean_str += s + " "
        lst_concat.append(clean_str)
    # print("spacy predicting", datetime.now())
    docs = list(nlp.pipe(lst_concat))
    lst_spacy = []  # list of floats for sentiment scores
    # print("spacy labels appending to lst_spacy", datetime.now())
    for doc in docs:
        lst_spacy.append(doc._.polarity)

    return (lst_flair, lst_spacy, lst_concat, texts)



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

    max_items = 100  # current speed 2min for getting 1000 tweets and analyze them using 2 models

    lst_flair, lst_spacy, lst_concat, texts = get_data(bearer_token, query, tweet_fields, max_items)
    # print("len of get_data output", len(lst_flair))
    # for i in range(len(lst_flair)):
    #     print(lst_flair[i], lst_spacy[i], lst_concat[i], texts[i])
    


if __name__ == "__main__":
    main()

# remove all retweets
# add max_item check
# set threshold e.g. -0.9 (later evaluation set etc.)
# [0, -0.3, -0.95] -> [0,0,1] i.e. 1 is negative text
# negative rate: count(1)/len(output_array)
# spot user: {user1: [#of tweets, #of retweets, #of likes], user2: [5, 100, 20]}

# TODO:
# sort neg_users by influence or number of retweets/likes
# each for-loop iteration now gives around 30 original tweets, we want to call get_score with a list of texts = 128 
# to optimize model efficiency/ batch size

# Does tweet_count include the user's replies? Shall use number of followers, not number of tweets?
# later: search thru specific toxic users identified
# Twitter Developer provides sentiment analysis for tweets
