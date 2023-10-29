from django.conf import settings
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS

real_path = settings.MEDIA_ROOT + "//" + 'RealNews.csv'
real = pd.read_csv(real_path, nrows=100)
fake_path = settings.MEDIA_ROOT + "//" + 'FakeNews.csv'
fake = pd.read_csv(fake_path, nrows=100)
# Create Target based on Real and Fake data
real['Category'] = 1
fake['Category'] = 0
dataset = pd.concat([real, fake]).reset_index(drop=True)
dataset['final_text'] = dataset['title'] + dataset['text']
dataset['final_text'].head()
dataset['Category'].value_counts()
dataset[['Category','subject','final_text']].groupby(['Category','subject']).count()
porter_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_text = []
lemmatized_text = []
final_text_result = []
for text in dataset['final_text']:
    result = re.sub('[^a-zA-Z]', ' ', text)
    result = result.lower()
    result = result.split()
    result = [r for r in result if r not in set(stopwords.words('english'))]
    final_text_result.append(" ".join(result))
    stemmed_result = [porter_stemmer.stem(r) for r in result]
    stemmed_text.append(" ".join(stemmed_result))
    lemmatized_result = [lemmatizer.lemmatize(r) for r in result]
    lemmatized_text.append(" ".join(lemmatized_result))

def get_prediction(vectorizer, classifier, X_train, X_test, y_train, y_test):
    pipe = Pipeline([('vector', vectorizer),
                    ('model', classifier)])
    model = pipe.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred),2)
    cl_report = classification_report(y_test, y_pred, output_dict=True)
    print("Accuarcy: {}".format(acc))
    print("Classification Report: \n", cl_report)
    return acc,cl_report


print("******USING STEMMED TEXT********")
X_train, X_test, y_train, y_test = train_test_split(stemmed_text, dataset['Category'], test_size = 0.3, random_state= 0)

def proces_real_or_fake_dataset():
    classifiers = [LogisticRegression(), LinearSVC(), MultinomialNB(), KNeighborsClassifier(n_neighbors=5), RandomForestClassifier()]
    rs = []
    for classifier in classifiers:
        print("\n\n", classifier)
        print("***********Usng Count Vectorizer****************")
        acc, cl_report = get_prediction(CountVectorizer(), classifier, X_train, X_test, y_train, y_test)
        cl_report = pd.DataFrame(cl_report).transpose()
        cl_report = pd.DataFrame(cl_report)
        d = {'acc': acc, 'cl_report':cl_report.to_html,'clssifier': classifier}
        rs.append(d)
        # print("***********Usng TFIDF Vectorizer****************")
        #get_prediction(TfidfVectorizer(), classifier, X_train, X_test, y_train, y_test)
    return rs