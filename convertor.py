import pandas as pd
import os
from tika import parser
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
def pdf_to_csv(fileloaction):
    df = pd.DataFrame(columns = ['CandidateID', 'resume', 'cleaned_resume'])
    for dirname, _, filenames in os.walk('./'+fileloaction):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            raw = parser.from_file(path)
            df = df.append({'CandidateID' : filename[:-4], 'resume' : " ".join(raw['content'].strip().split('\n')[1:]), 'cleaned_resume' : ""}, 
                    ignore_index = True)
    return df

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def del_resume(data):
    data.drop(columns=['resume'],inplace=True)
    return data


def data_corrector(data):
    #Converting the data to pdf
    data = pdf_to_csv('trainResumes')
    #Cleaning data
    data['cleaned_resume'] = data.resume.apply(lambda x: cleanResume(x))
    data = del_resume(data)
    #Importing target variable and joining it with data
    train = pd.read_csv("train.csv")
    train.set_index('CandidateID',inplace=True)
    data.set_index("CandidateID",inplace=True)
    train = train.join(data)
    #Scalling the features
    test_feature = data
    feature = train.cleaned_resume
    vectorizer = TfidfVectorizer()
    v_feature = vectorizer.fit_transform(feature).toarray()
    test_feature = vectorizer.transform(test_feature).toarray()
    return test_feature
