from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import pandas as pd
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import Parallel, delayed 
import joblib

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
svc = SVC(kernel="rbf", gamma=0.5, C=1.0)

lem = WordNetLemmatizer()

data = pd.read_csv("hate_speech.csv", sep=",")[["text", "label"]]
data["label"] = LabelEncoder().fit_transform(data["label"])

print("starting to normalize data !")
for i in range(len(data)):
    data["text"][i] = " ".join([lem.lemmatize(word) for word in data["text"][i].replace(",", "").replace("!", "").replace(".", "").lower().strip().split(" ") if [l.isdigit() for l in word] and len(word) > 2])
print("normalizing data done !")

X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# for i in range(len(X_train)):
#     vectorizer = TfidfVectorizer(stop_words='english')

# for i in range(len(X_test)):
#     vectorizer = TfidfVectorizer(stop_words='english')
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("starting training !")
svc_model = SVC(kernel='poly', degree=1, C=0.5, probability = True)
svc_model.fit(X_train_tfidf, y_train)
# k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(svc_model, X_train_tfidf, y_train, cv=k_fold, scoring='accuracy')
print("training done !")

y_pred = svc_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# joblib.dump(svc_model, "svc_model.pkl")
# joblib.dump(vectorizer, "vectorizer.pkl")
