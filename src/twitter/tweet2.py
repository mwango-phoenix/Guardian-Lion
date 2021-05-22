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
    # url = "https://api.twitter.com/2/tweets/search/all?query={}&max_results=100&{}&expansions=author_id&user.fields=public_metrics".format(query, tweet_fields)
    url += "&start_time=2021-05-15T23:00:00.000Z&end_time=2021-05-21T00:00:00.000Z"  # change the dates, up to 7 days ago
    # # need to change project-level: https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all

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


# fix:
# def get_tweet_by_id(original_tweet_id):
#     # find the original tweet by id
#     url = "https://api.twitter.com/2/tweets/{}?tweet.fields=text,author_id".format(original_tweet_id)
#     original_tweet_res = connect_to_endpoint(url, headers)


# in batches, get Tweets (json_response) and feed into clean_up & output scores from 2 models
def get_data(bearer_token, query, tweet_fields, max_items):  # -> Tuple[List[float]]
    # print("entering get_data", datetime.now())
    next_token = ''
    total_items = 0
    data = {'data' : []}
    filename = get_file_name(query)
    all_texts = []
    while total_items < max_items:
        # print("entering while loop/api", datetime.now())
        url = create_url(query, tweet_fields, next_token)
        headers = create_headers(bearer_token)
        json_response = connect_to_endpoint(url, headers)
        lst_texts = []
        lst_i = []
        # print(len(json_response['data']), len(json_response['includes']['users']))
        for i in range(len(json_response['data'])):
            # TODO: var = json_response['data'][i]
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

        # print("*********done", datetime.now())

        total_items += len(lst_texts)
        for item in lst_i:
            json_response['data'].remove(item)
        all_texts.extend(lst_texts)
        data['data'] = data['data'] + json_response['data']  # this json_response includes original tweets only

        if 'next_token' not in json_response['meta']:
            break
        next_token = json_response['meta']['next_token']

    # get sentiment score in batches
    flair_scores = []
    spacy_scores = []
    cleaned_texts = []
    raw_texts = []
    batch_size = 128
    i = 0
    while i < total_items:
        batch_scores = get_score(all_texts[i:(min(i + batch_size, len(all_texts)))])
        flair_scores.extend(batch_scores[0])
        spacy_scores.extend(batch_scores[1])
        cleaned_texts.extend(batch_scores[2])
        raw_texts.extend(batch_scores[3])
        i += batch_size
    
    idx_arr = get_idx_arr(flair_scores, -0.96)  # get array of idx of negative texts
    neg_rate = len(idx_arr)/len(flair_scores)
    print("neg_rate:", neg_rate)

    neg_users = find_neg_users(idx_arr, data['data'])
    print("neg_users:", neg_users)  # sorted

    with open('./new.json', "w") as f:
        json.dump({'count': total_items, 'neg_rate_of_scrapped': neg_rate, 'neg_users': neg_users, 'data': data['data']}, 
                  f, indent=4, sort_keys=True)

    return (flair_scores, spacy_scores, cleaned_texts, raw_texts)

def find_neg_users(idx_arr, data):
    neg_users = {}
    for idx in idx_arr:
        tweet = data[idx]
        user = tweet['author_id']
        if user not in neg_users:
            # [username, # of total tweets, # of followers, # of retweets, #of likes, # of neg tweets found, texts of neg tweets found]
            # aybe change to class object: attribute as key, attribute value as value
            # turning into json from object: obj.__dict__
            neg_users[user] = {'username': tweet['username'],
                               'tweet_count': tweet['tweet_count'], 
                               'followers_count': tweet['followers_count'],
                               'retweet_count': 0, 'like_count': 0, 'found_count': 0,
                               'texts_arr': []}

        neg_users[user]['retweet_count'] += tweet['public_metrics']['retweet_count']
        neg_users[user]['like_count'] += tweet['public_metrics']['like_count']
        neg_users[user]['found_count'] += 1
        neg_users[user]['texts_arr'].append(tweet['text'])

        # sort by number of likes and number of followers
        neg_users = dict(sorted(neg_users.items(), key=lambda item: (item[1]['like_count'], item[1]['followers_count']), reverse=True))

    return neg_users

# classify
def get_idx_arr(flair_scores, threshold):
    raw_scores = np.array(flair_scores)
    idx_arr = np.nonzero(raw_scores < threshold)
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
    

    # using Flair for sentiment analysis
    print("flair making sentences", datetime.now())
    lst_sentences = []
    for lst in lst_cleaned:
        # print("making sentence")
        if len(lst) != 0:
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

    tweet_fields = "tweet.fields=text,author_id,lang,created_at,referenced_tweets,in_reply_to_user_id,conversation_id,possibly_sensitive,public_metrics"  
    # search term
    query = 'lang:en "china virus"'

    max_items = 200

    get_data(bearer_token, query, tweet_fields, max_items)
    

# '1026576700610621441': ['LangmanVince', 35658, 16826, 18, 40, 1, ['The left lied about \nRussian Collusion \nImpeachment 1&amp;2\nChina Virus \nPeaceful protest \nInsurrection \nThe 2020 election \nAmerica is racist \nThe police are racist \nSo I can lie about getting a vaccine when I walk into Walmart without a mask']]

if __name__ == "__main__":
    main()


# Done:
# remove all retweets
# add max_item check
# set threshold e.g. -0.9 (later evaluation set etc.)
# [0, -0.3, -0.95] -> [0,0,1] i.e. 1 is negative text
# negative rate: count(1)/len(output_array)
# spot user: {user1: [#of tweets, #of retweets, #of likes], user2: [5, 100, 20]}
# sort neg_users by influence or number of retweets/likes
# optimize model efficiency/ batch size: each while-loop iteration gives around 30 original tweets, 
#   we want to call get_score with a list of texts = 128 
# Does tweet_count include the user's replies? Shall use number of followers, not number of tweets?

# TODO:
# design some metrics as a combination of likes, folowers etc.
# plot using neg_users: e.g. followers vs. likes of neg text
# np.hist(arr_likes_out_of_all_neg_texts)
# function for sorting by parameter
# later: search thru specific toxic users identified
# Twitter Developer provides sentiment analysis for tweets