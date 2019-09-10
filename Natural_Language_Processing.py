# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:34:20 2019

@author: Sayak
"""
#Natural Language Processing

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

#cleaning text 
import re
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
corpus = []
for i in range (1000):
    review = re.sub('[^a-zA-Z]',' ', dataset["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()      #stemming - taking root of the word ie, liking,liked changed to like
    #l = []
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]     #removing the unnecessary words like prepositions, articles, me, I, themselves using NLTK library
    '''for i in review:
        if i not in set(stopwords.words('english')):
            word = ps.stem(i)
            l.append(word)
    print (l)
    '''
    review = " ".join(review)
    corpus.append(review)

#create the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#splitting the dataset into test-train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)  

'''#feature scaling   #not needed as matrix contains mostly 0's and 1's and a few 2's, 3's
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)'''

#fit classifier Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

'''#fitting the classifier DecisionTree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)'''

'''#fitting the classifier RandomForest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)'''

#predicting the results
y_pred = classifier.predict(X_test)
 
#Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#(74+68)/200
#(55+91)/200    -> maximum accuracy for Naive Bayes of 73% so keeping it and neglacting the others 
#(87+56)/200