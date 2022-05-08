import tweepy
import secrets #use your own secrets
import re
#import nltk #just do this once
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

client = tweepy.Client(BEARER_TOKEN)

doc1 = client.search_recent_tweets(query='#covid #vaccine -is:retweet lang:en',max_results=100,tweet_fields=['lang'])

stop = set(stopwords.words('english'))
def remove_stop(text):
    text = text.lower().split()
    text = [word for word in text if word not in stop]
    return ' '.join(text)
  
ps = PorterStemmer()
processed_tweets = []
for tweet in doc1.data:
    prep_tweet = re.sub(r'\n',' ',tweet['text'].lower())
    prep_tweet = re.sub(r'(w+://S+)','',prep_tweet)#https?[:/\.a-zA-Z]+
    prep_tweet = re.sub(r'[|.,!\[\]?$:;\-+=\'()%...\"/@#_]+','',prep_tweet)
    prep_tweet = re.sub(r'http\w+','',prep_tweet) #still seeing urls for some reason
    prep_tweet = re.sub(r'&amp','',prep_tweet)
    prep_tweet = re.sub(r'\d+','',prep_tweet)
    prep_tweet = remove_stop(prep_tweet)
    prep_tweet = word_tokenize(prep_tweet)
    prep_tweet = [ps.stem(word) for word in prep_tweet]
    processed_tweets.append(prep_tweet)
    
#this is just out of curiousity; see what words we are getting etc 
wordset = set()
for tweet in processed_tweets:
    for word in tweet:#.split(' '):
        wordset.add(word)
print(len(wordset))


    
