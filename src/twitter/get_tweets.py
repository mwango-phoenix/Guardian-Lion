import requests
import os
import json
from datetime import date


# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'


def auth():
    return os.environ.get("BEARER_TOKEN")


def create_url(query, tweet_fields, next_token):
    url = "https://api.twitter.com/2/tweets/search/recent?query={}&max_results=100&{}".format(query, tweet_fields)
    if next_token != '':
        url = url + '&next_token={}'.format(next_token)
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )

    return response.json()


def get_file_name(query, token=''):
    return '../../data/{}_{}_{}.json'.format(date.today().strftime("%Y_%m_%d"), query, token)

def get_data(bearer_token, query, tweet_fields, max_items):
    next_token = ''
    total_items = 0
    data = {'data' : []}
    filename = get_file_name(query)
    while total_items < max_items:
        url = create_url(query, tweet_fields, next_token)
        headers = create_headers(bearer_token)
        json_response = connect_to_endpoint(url, headers)
        total_items = total_items + len(json_response['data'])
        data['data'] = data['data'] + json_response['data']
        print(json.dumps(json_response, indent=4, sort_keys=True))
        with open(filename, "w") as f:
            f.write(json.dumps(data, indent=4, sort_keys=True))

        filename_token = get_file_name(query, next_token)
        with open(filename_token, "w") as f:
            f.write(json.dumps(json_response, indent=4, sort_keys=True))

        next_token = json_response['meta']['next_token']


def main():
    bearer_token = auth()

    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    tweet_fields = "tweet.fields=tex"

    # search term
    query = "china japan"

    max_items = 10000


    get_data(bearer_token, query, tweet_fields, max_items)


if __name__ == "__main__":
    main()