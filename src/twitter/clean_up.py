# clean up raw tweet using spacy and feed into sentiment analyzers, outputing scores
# this file incorporates Michelle's work in preprocess.py

import flair  # sentiment analysis
import spacy
from spacymoji import Emoji
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacytextblob.spacytextblob import SpacyTextBlob  # sentiment analysis
import contractions

# pip3 install flair
# pip3 install -U pip setuptools wheel
# pip3 install -U spacy
# python3 -m spacy download en_core_web_sm
# pip3 install spacymoji
# pip3 install contractions
# pip3 install spacytextblob

# sample tweets
text1 = "Terming the variant of the Coronavirus as an Indian variant -  is a racist slur by @WHO. Indian variant in a report by TOIIndiaNews. For over a year people were termed racist for calling coronavirus as China virus but now being racists against India is normal by DrTedros"
text2 = "I'm not watching @JoeBiden , occupant of the WH but I can tell you what he's going to say. \n\n1. Biden's fault, everything\n2. Come on illegals. Front line for you, Dems need your votes you get money b4 Americans.\n3. Get China Virus vaccine\n4. I can't fix shit\n5. Puddin time?"
text3 = "@ImPaulRNelson @catturd2 @Acosta When will China be held accountable for creating and releasing the China virus?"
text4 = "@MayorJenny @komonews Seems silly for healthy young people whose risk of dying from covid approaches zero, to take an experimental vaccine with unknown long-term side effects. \n\nI think I’ll wait a while until maybe it’s actually an FDA approved vaccine 👍\n\nThank you for offering though!"
text5 = "@MarshaBlackburn @WHO @TECRO_USA @MOFA_Taiwan Senator Blackburn ...question 🙋🏽\u200d♂️\n\nWhat is the status of the bill you bragged about so much to enable us to sue China for the “China Virus”??\n\nP.S. No doubt Taiwan doesn’t want this batshit lady speaking for them."


# expand contraction
raw_sentence = contractions.fix(text1)

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
doc = nlp(raw_sentence)
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


# using Flair for sentiment analysis
sentence = flair.data.Sentence(clean_sentence_lst)
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
flair_sentiment.predict(sentence)
total_sentiment = sentence.labels
print("sentiment", total_sentiment)


# using SpaCy's Textblob for sentiment analysis
# https://spacytextblob.netlify.app/docs/example, can be for multiple texts
nlp.add_pipe('spacytextblob')
doc = nlp(text1)
print("sentiment", doc._.polarity)