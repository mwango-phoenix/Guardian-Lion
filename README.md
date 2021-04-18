# Super Nova Machine Learning Group

## Introduction

## Team
See [Team members](./team/team.md)

## Meeting Minutes
- [20210328 Kickoff](./meetingminutes/20210328.md)
- [20210404 Project selection](./meetingminutes/20210404.md)

## Resources 
- [Git cheat sheet](./resources/git.md)
- [[Deep learning] How to build an emotional chatbot](https://towardsdatascience.com/deep-learning-how-to-build-an-emotional-chatbot-part-1-bert-sentiment-predictor-3deebdb7ea30)
- [Convert Twitter ID to User ID](https://tweeterid.com/)

## Get twitter data

### API Key

API Key: RJLurQA1nGWvOUIngDY3UxDHy
API Secret Key: 0kbblMYMozyeQ1DpmDGOYpr9dMzuNVwHefhD9mRVre0qvpCvd1
Bearer Token: AAAAAAAAAAAAAAAAAAAAAPHwOQEAAAAAWYxm9Ej2VmMX3WEnWmdCqVM2%2FGo%3DD7agAk1Zvr4GXSD4BxT3m6KxPlVyXw0rS5y2pqvgpdLuSUIBjd

### Run the script

```shell script
export BEARER_TOKEN=Bearer Token
python3 ./src/twitter/get_tweets.py
```

### Convert json to CSV:

```
cat 2021_04_11_china_.json | jq -r '.data[] | {author_id,id,text}' | jq -s | jq -r '(map(keys) | add | unique) as $cols | map(. as $row | $cols | map($row[.])) as $rows | $cols, $rows[] | @csv' > 2021_04_11_china.csv
```

## Run jupyter

### Installation

```shell script
pip3 install jupyterlab
```

### Run a local jupyter server
Run the command below and a web browser will be started.
```shell script
jupyter-lab
```

There is a script under src/jupyter. You can also create a new notebook.

Commonly used commands for jupyter can be found [here](http://maxmelnick.com/2016/04/19/python-beginner-tips-and-tricks.html).

