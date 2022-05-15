from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
 
 
stop = set(stopwords.words('english'))
lm = WordNetLemmatizer()
def remove_stop(text):
    text = text.lower().split()
    text = [word for word in text if word not in stop]
    return ' '.join(text)

def get_processed_data(data):
    processed_tweets = []
    for tweet in data:
        prep_tweet = re.sub(r'(\w+://\S+)','',tweet.lower()) #remove links
        prep_tweet = re.sub(r'&amp','',prep_tweet) #I saw a number of these; dont want them showing up
        prep_tweet = re.sub(r'[^a-zA-Z0-9\s]','',prep_tweet) #remove emojis and punctuation
        prep_tweet = remove_stop(prep_tweet) #remove the stopwords - maybe this is too early
        prep_tweet = ' '.join([lm.lemmatize(word) for word in prep_tweet.split(' ')]) #lemmatise the data
        prep_tweet = prep_tweet.replace(' nh ',' nhs ') #I noticed this was being lemmatised, even though it is an important term
        processed_tweets.append(prep_tweet)
    return processed_tweets

def make_bigrams(texts): #Train the model on our data, then return words based on our data. Iunno I dont make the rules here
    bigram = Phrases(texts, min_count=5, threshold=100)
    bigram_mod = Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]

def popular_word_culler(doc_list): #doc_list is a list of strings
    cvec = CountVectorizer(analyzer='word',       
                                 min_df=10,
                                 token_pattern='[\w]{3,}',
                                )
    bow = cvec.fit_transform(doc_list)
    bow_df = pd.DataFrame(bow.toarray(),columns=cvec.get_feature_names_out())
    occurs = dict(zip([bow_df.T.iloc[x].name for x in range(len(bow_df.T))],
                      [len(bow_df) - bow_df.T.iloc[x].to_list().count(0) for x in range(len(bow_df.T))]))
    wanted_words = ['biden', 'boosted', 'booster', 'case', 
      'child', 'country', 'death', 'died', 
      'everyone', 'first', 'fully', 'good', 
      'health', 'kid', 'know', 'long', 
      'mandate', 'mask',  'new', 'pandemic', 
      'pfizer', 'rate', 'realcandaceo', 
      'risk', 'sorry', 'think', 'trump', 'work']
    unwanted_words = [word for word in occurs.keys() if occurs[word] > 600 and word not in wanted_words]
    new_tweets = []
    for tweet in doc_list:
        for word in unwanted_words:
            tweet = tweet.replace(f' {word} ',' ')
        new_tweets.append(tweet)
    new_tweets_list = [[word for word in tweet.split()] for tweet in new_tweets]
    return new_tweets, new_tweets_list

def short_word_culler(doc_list): #doc_list is a list of strings
    long_word_tweets = []
    for tweet in doc_list:
        long_word_tweets.append(
            ' '.join([word for word in tweet.split() if len(word)>3])
        )

    long_word_tweets_list = [[word for word in tweet.split()] for tweet in long_word_tweets]
    return long_word_tweets, long_word_tweets_list

def number_culler(doc_list): #doc_list is a list of strings
    a=[re.sub('\b\d+\b','',doc) for doc in doc_list]
    return a, [[word for word in doc.split()] for doc in a]

