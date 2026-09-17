from contractions import fix
import re
from emoji import demojize
import string
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# Download Bundels 
nltk.download("punkt")
nltk.download("stopwords")


# Decalration of stopwords and SnowballStemmer

stop_word = set(stopwords.words("english"))
stem  = SnowballStemmer(language="english")


def data_process(text):
    # convert String into lower
    text = text.lower()
    # fix contraction 
    text = fix(text)
    # remove links from string 
    text = re.sub(r"http\S+","",text)
    # remove html tags from string 
    text = re.sub(r"<.*?>","",text)
    # filtering Emojis
    text = demojize(text,language="en",delimiters=(" "," ")) 
    text = text.replace("_"," ")
    # remove special character
    text = text.translate(str.maketrans(" " ," " ,string.punctuation))
    # text = re.sub(r"[^a-zA-Z]\s","",text)
    # convert string into tokens
    text = word_tokenize(text)
    # Remove stop_words
    text = [word for word in text if word not in stop_word]
    # convert into original form 
    text = [stem.stem(word) for word in text]
    # convert into string 
    text = " ".join(text)
    return text
