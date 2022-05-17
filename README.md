# ICT202-Twitter-Project
A project to scrape twitter for tweets about covid19 vaccinations IOT learn data processing 
Project is to be written in python using the twitter API and libraries including tweepy and searchtweet. 
Main outcomes are evaluatition of different methods of Topic Modelling (Unsupervised learning to determine what other topics are brought up when discussing vaccination), and the data preprocessing used.
Word cloud tracking 100 most popular words for each discovered topic
Data is to be correctly preprocessed, involving stemming and removing stop words. 
Evaluation of different techniques of feature extraction and modelling. 


Planned Outcomes:
Process used to clean data
Graphs of evaluation of topic/cluster numebrs used on different models
Wordcloud of common words in each discovered classification

Marking Guide Plan:
Data Collection (10%): This should be hard to get wrong; so long as a stream of tweets is collected through the API we should get full marks here
Data preprocessing (20%): 
Different models work better with data processed in different ways
Clean data of unwanted characters (punctuation and &amp style bits
Remove hyperlinks 
Remove stop words
Form n grams
remove single words that do not look like they will contribute much ('vaccine_hesitance' might be good, but 'vaccine' will be common to most docs)
remove single numbers (keep covid_19, remove 19 by itself)
Exploratory Analysis (10%): Unsure of what is meant by this, maybe this is determining how many clusters 
Feature Extraction (20%): TFIDF and BoW I think 
Model building (20%): Discuss the parameters used to train models (number of clusters, random state freezing, learning rate, any other)
Performance Eval (10%): Discuss the improvements of results as data preprocessing was optimised
Discuss the performance of different models according to the eyeball method. 
Report structure etc (10%): 
