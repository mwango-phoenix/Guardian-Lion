import flair

# pip3 install flair
# pip3 install -U pip setuptools wheel
# pip3 install -U spacy
# python3 -m spacy download en_core_web_sm
# pip3 install spacymoji

text1 = "Terming the variant of the Coronavirus as an Indian variant -  is a racist slur by @WHO. Indian variant in a report by TOIIndiaNews. For over a year people were termed racist for calling coronavirus as China virus but now being racists against India is normal by DrTedros"
# TO remove @... name tag (not tokenized as separate to name; need manual removal see https://stackoverflow.com/questions/62141647/remove-emojis-and-users-from-a-list-in-python-and-punctuation-nlp-problem-and)
#   names are relatively neutral and some usernames don't make sense, so we'll remove the whole name tag
# TO remove punctuations using spacy token.is_punct
# affecting the score: url, @, punctuations

# text2 = "I'm not watching @JoeBiden , occupant of the WH but I can tell you what he's going to say. \n\n1. Trump's fault, everything\ud83d\ude44\n2. Come on illegals. Front line for you, Dems need your votes &amp; you get money b4 Americans.\n3. Get China Virus vaccine\n4. I can't fix shit\n5. Puddin time?"
text2 = "I'm not watching @Trump , occupant of the WH but I can tell you what he's going to say. \n\n1. Biden's fault, everything\n2. Come on illegals. Front line for you, Dems need your votes you get money b4 Americans.\n3. Get China Virus vaccine\n4. I can't fix shit\n5. Puddin time?"
# use the original json, not json dump's output display (\ud83d unrecognized unicode error)
# \n ignored by tokenizer automatically
# with a bad emoji, it's evaluated as less negative
# TO remove emojis, first tokenize with spacymoji (https://spacy.io/universe/project/spacymoji) and then clean up & remove other special chars see https://stackoverflow.com/questions/62141647/remove-emojis-and-users-from-a-list-in-python-and-punctuation-nlp-problem-and
# TO remove &amp;
# TO remove numbers (another example: "wasted 1 million dollars" is more -ve than "wasted 100 million dollars")


text3 = "@ImPaulRNelson @catturd2 @Acosta When will China be held accountable for creating and releasing the China virus?"
# TO remove stop words e.g. the, using spacy token.is_stop
# TO lemmatize/stem e.g. releasing and release gives diff scores
# w/ @s -> -0.96 whereas without @s -> -0.673
# TO remove @ for names that don't make sense? yes

# text4 = "@MayorJenny @komonews Seems silly for healthy young people whose risk of dying from covid approaches zero, to take an experimental vaccine with unknown long-term side effects. \n\nI think I\u2019ll wait a while until maybe it\u2019s actually an FDA approved vaccine \ud83d\udc4d\n\nThank you for offering though!"
text4 = "@MayorJenny @komonews Seems silly for healthy young people whose risk of dying from covid approaches zero, to take an experimental vaccine with unknown long-term side effects. \n\nI think I’ll wait a while until maybe it’s actually an FDA approved vaccine 👍\n\nThank you for offering though!"
# Upper case doesn't affect score
# TO expand contractions (e.g. "I'll" and "I will" give diff scores) using spacy, see https://gist.github.com/widiger-anna/deefac010da426911381c118a97fc23f (need to find new link)

# text5 = "@MarshaBlackburn @WHO @TECRO_USA @MOFA_Taiwan Senator Blackburn ...question \ud83d\ude4b\ud83c\udffd\u200d\u2642\ufe0f\n\nWhat is the status of the bill you bragged about so much to enable us to sue China for the \u201cChina Virus\u201d??\n\nP.S. No doubt Taiwan doesn\u2019t want this batshit lady speaking for them. https://t.co/Xr7xrQY0ps"
text5 = "@MarshaBlackburn @WHO @TECRO_USA @MOFA_Taiwan Senator Blackburn ...question 🙋🏽\u200d♂️\n\nWhat is the status of the bill you bragged about so much to enable us to sue China for the “China Virus”??\n\nP.S. No doubt Taiwan doesn’t want this batshit lady speaking for them."
# \u200d is tokenized but doesn't affect score so don't need to remove
# TO remove special char e.g. ♂️
# using token attributes: https://spacy.io/api/token

raw_sentence = text2

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
spacy_tokenizer = flair.tokenization.SpacyTokenizer('en_core_web_sm')
# clean up raw data using spacy
# 1. tokenization
# with spacymoji added to pipeline
# keep hyphen/one word, see https://spacy.io/usage/linguistic-features#tokenization

# -> List[Token]

# 2. removing punctuation/ url/ special chars/ numbers/ stopwords; remove name tags

# -> List[Token]

# to check for special chars, if it's emoji or not:
# import spacy
# from spacymoji import Emoji
# nlp = spacy.load("en_core_web_sm")
# emoji = Emoji(nlp)
# nlp.add_pipe(emoji, first=True)
# doc = nlp("♂️")
# for token in doc:
#     print(token._.is_emoji)

# 3. expand contraction

# 4. lemmatization

# -> List[str]


# sentence = flair.data.Sentence(clean_sentence_lst)
sentence = flair.data.Sentence(raw_sentence, use_tokenizer=spacy_tokenizer)
# sentence = flair.data.Sentence(['with', 'negative', 'long', 'term', 'effects'])

flair_sentiment.predict(sentence)
total_sentiment = sentence.labels
print(sentence)
# for token in sentence:
#     print(token)
print(total_sentiment, sentence.labels[-1].score)
# for label in sentence.labels:
#     print(label.value, label.score)


# print(sentence.get_embedding())  # ?? no embedding? switch to en_core_web_lg?

# ?? How to get the score for each token? unable to track from the code for predict()
# for tok in sentence.tokens:
#     print(tok)
#     print(tok._embeddings)
#     print(tok.tags_proba_dist)