import pandas as pd
import numpy as np
import os
import convertor
from sklearn.feature_extraction.text import TfidfVectorizer
#Converting the data to pdf
data = convertor.pdf_to_csv('trainResumes')
#Cleaning data
data['cleaned_resume'] = data.resume.apply(lambda x: convertor.cleanResume(x))
data = convertor.del_resume(data)
#Importing target variable and joining it with data
train = pd.read_csv("train.csv")
train.set_index('CandidateID',inplace=True)
data.set_index("CandidateID",inplace=True)
train = train.join(data)
#Importing and cleaning test data
test = convertor.pdf_to_csv('testResumes')
test['cleaned_resume'] = test.resume.apply(lambda x: convertor.cleanResume(x))
test = convertor.del_resume(test)
#Scalling the features
test_feature = test['cleaned_resume']
feature = train.cleaned_resume
vectorizer = TfidfVectorizer()
v_feature = vectorizer.fit_transform(feature).toarray()
test_feature = vectorizer.transform(test_feature).toarray()
labels = []
for i in train['Match Percentage']:
    labels.append(i)
labels = np.array(labels)
labels = labels.reshape(-1,1)
#Creating a model
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
clf.fit(v_feature,labels)
#Saving the model
from joblib import dump, load
dump(clf, 'src.joblib')