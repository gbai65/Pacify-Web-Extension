from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed 
import joblib 
from nltk.corpus import stopwords
import nltk
import numpy as np

nltk.download('stopwords')
lem = WordNetLemmatizer()
f = open("svc_model.pkl", "rb")
loaded_model = joblib.load(f) 
v = open("vectorizer.pkl", "rb")
loaded_v = joblib.load(v)
def remove_hate(data):
        data = " ".join([lem.lemmatize(word) for word in data.replace(",", "").replace("!", "").replace(".", "").lower().strip().split(" ") if [l.isdigit() for l in word] and len(word) > 2 and word not in stopwords.words('english')])

        X_test_tfidf = loaded_v.transform([data])
        result = loaded_model.predict(X_test_tfidf)
        # with open("results.txt","w") as f:
        #     f.write(result.tos)
        if result == [0]:
            print("hate")
        else:
            print("nothate")
        return result

print(remove_hate("i love men"))