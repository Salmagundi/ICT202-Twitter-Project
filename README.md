# ICT202-Twitter-Project
A project to scrape twitter for tweets about covid19 vaccinations IOT learn data processing 
Project is to be written in python using the twitter API and libraries including tweepy and searchtweet. 
Main outcomes appear to be Topic Modelling (Unsupervised learning to determine what other topics are brought up when discussing vaccination)
Word cloud tracking 100 most popular words for each discovered topic
Data is to be correctly preprocessed, involving stemming and removing stop words. 
Evaluation of different techniques of feature extraction and modelling. 


Planned Outcomes:
Wordcloud of all terms after text processing
Wordcloud after TF-IDF
Plot of tweets by unsupervised classification
Wordcloud of common words in each discovered classification

Marking Guide Plan:
Data Collection (10%): This should be hard to get wrong; so long as a stream of tweets is collected through the API we should get full marks here
Data preprocessing (20%): This is more involved, but still not a problem and already mostly solved withthe curernt state of things. 
  Data is collected, punctuation and other noise (urls, numbers, etc) are removed 
  Stemming/Lemminisation of words. Stemming not ideal for word clouds. 
Exploratory Analysis (10%): This is probably a more complicated bit, but is not graded highly. 
  Use TF IDF to find important terms, and cluster based on the top n terms. 
  https://towardsdatascience.com/applying-machine-learning-to-classify-an-unsupervised-text-document-e7bb6265f52
  Above link may be useful 
  sklearn.feature_extraction.text.TfidfVectorizer
  word2vec is another recommended solution 
  https://radimrehurek.com/gensim/ - gensim
  https://towardsdatascience.com/2-latent-methods-for-dimension-reduction-and-topic-modeling-20ff6d7d547 LSI topic modelling 
Feature Extraction (20%): Clearly a more important bit. This should be the unsupervised learning I think, and will be creating the bag of words, discovering topics, and deonstrating findings. 
Model building (20%): Feels like the above, but instead of demonstrating results, this is probably the driver behind the scenes that makes it work. The python files etc. 
Performance Eval (10%): Discuss the sparce nature of BoW and the improvements that n-grams provide maybe? Who knows. 
Report structure etc (10%): This will naturally occur. 
