import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# NLP libraries to clean the text data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
# Vectorization technique TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# For Splitting the dataset
from sklearn.model_selection import train_test_split
# Model libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from django.conf import settings
# Accuracy measuring library
from sklearn.metrics import accuracy_score

path = settings.MEDIA_ROOT + "//" + 'jruvika.csv'
data = pd.read_csv(path, nrows=10000)
df = data.copy()  # Creating a copy of my data, I will be working on this Dataframe
df['Body'] = df['Body'].fillna('')
df.isnull().sum()
df['News'] = df['Headline'] + df['Body']
features_dropped = ['URLs', 'Headline', 'Body']
df = df.drop(features_dropped, axis=1)
ps = PorterStemmer()


def wordopt(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text


df['News'] = df['News'].apply(wordopt)  # Applying the text processing techniques onto every row data
X = df['News']
Y = df['Label']

# Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


def process_SVM():
    # 2. Support Vector Machine(SVM) - SVM works relatively well when there is a clear margin of separation between classes.
    svm_model = SVC(kernel='linear')
    # Fitting training set to the model
    svm_model.fit(xv_train, y_train)
    # Predicting the test set results based on the model
    svm_y_pred = svm_model.predict(xv_test)
    # Calculate the accuracy score of this model
    score = accuracy_score(y_test, svm_y_pred)
    svm_report = classification_report(y_test, svm_y_pred, output_dict=True)
    print('Accuracy of SVM model is ', score)
    return score, svm_report


def process_LogisticRegression():
    # 1. Logistic Regression - used because this model is best suited for binary classification
    LR_model = LogisticRegression()
    # Fitting training set to the model
    LR_model.fit(xv_train, y_train)
    # Predicting the test set results based on the model
    lr_y_pred = LR_model.predict(xv_test)
    # Calculate the accurracy of this model
    lg_acc = accuracy_score(y_test, lr_y_pred)
    lg_report = classification_report(y_test, lr_y_pred, output_dict=True)
    print('Accuracy of LR model is ', lg_acc)
    return lg_acc, lg_report


def process_randomForest():
    # 3. Random Forest Classifier
    RFC_model = RandomForestClassifier(random_state=0)
    # Fitting training set to the model
    RFC_model.fit(xv_train, y_train)
    # Predicting the test set results based on the model
    rfc_y_pred = RFC_model.predict(xv_test)
    # Calculate the accuracy score of this model
    rf_acc = accuracy_score(y_test, rfc_y_pred)
    rf_report = classification_report(y_test, rfc_y_pred, output_dict=True)
    print('Accuracy of RFC model is ', rf_acc)
    return rf_acc, rf_report



def process_naiveBayes():
    # 3. Naive Bayes
    nb_model = GaussianNB()
    # Fitting training set to the model
    nb_model.fit(xv_train.toarray(), y_train)
    # Predicting the test set results based on the model
    nb_y_pred = nb_model.predict(xv_test.toarray())
    # Calculate the accuracy score of this model
    nb_acc = accuracy_score(y_test, nb_y_pred)
    nb_report = classification_report(y_test, nb_y_pred, output_dict=True)
    print('Accuracy of Naive Bayes model is ', nb_acc)
    return nb_acc, nb_report



def process_knn():
    # 3. K Nearest Neighbour
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    # Fitting training set to the model
    knn_model.fit(xv_train, y_train)
    # Predicting the test set results based on the model
    knn_y_pred = knn_model.predict(xv_test)
    # Calculate the accuracy score of this model
    knn_acc = accuracy_score(y_test, knn_y_pred)
    knn_report = classification_report(y_test, knn_y_pred, output_dict=True)
    print('Accuracy of KNN model is ', knn_acc)
    return knn_acc, knn_report


def fake_news_det(news):
    svm_model = SVC(kernel='linear')
    svm_model.fit(xv_train, y_train)
    input_data = {"text": [news]}
    new_def_test = pd.DataFrame(input_data)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    # print(new_x_test)
    vectorized_input_data = vectorization.transform(new_x_test)
    prediction = svm_model.predict(vectorized_input_data)

    if prediction == 1:
       return "Not a Fake News"
    else:
        return "Fake News"