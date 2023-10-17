import contractions
import numpy as np
import pandas as pd
import string
from nltk import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pandarallel import pandarallel as pll
from nltk.stem import WordNetLemmatizer
import joblib
import os
from sklearn.metrics import matthews_corrcoef, classification_report, f1_score, recall_score, precision_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

pll.initialize(nb_workers=20, progress_bar=True)

# %%
punc = list(string.punctuation).copy()


def preprocess(review, punctuation, cont_func, lemmetizer, tokenizer):
    review = review.lower()
    review = review.strip()
    review = review.translate(str.maketrans('', '', "".join(punctuation)))
    review = review.translate(str.maketrans('', '', '0123456789'))
    review = cont_func(review)
    x = tokenizer(review)
    review = " ".join([lemmetizer(i, pos='v') for i in x])
    return review


# %%
path_large = "D:/Datasets/Amazon Reviews - Large/Amazon US Review/amazon_reviews_us_Wireless_v1_00.tsv"
data = pd.read_csv(path_large, sep='\t', on_bad_lines='skip', low_memory=False, usecols=['review_body', 'star_rating'])
data.columns = ['sentiment', 'review']
data.dropna(axis=0, inplace=True)
# %%
lemm = WordNetLemmatizer()
data['sentiment'] = data['sentiment'].map({'1': 0, '2': 0, '3': 0, '4': 1, '5': 1})
data['review'] = data['review'].parallel_apply(preprocess, args=(
    punc, contractions.fix, lemm.lemmatize, word_tokenize))
X = data['review']
Y = data['sentiment']
# %%
tfidf_file = "D:/PyCharm_Projects/Sentiment_Analysis/vectorizer/tfidf.vec"
if os.path.isfile(tfidf_file):
    vec = joblib.load(tfidf_file)
else:
    vec = TfidfVectorizer(stop_words='english', strip_accents='ascii', lowercase=False)
    vec.fit(X)
    joblib.dump(vec, tfidf_file)
X = vec.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=1)
# %%
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# %%

fit = False
saga_lr_file = 'D:/PyCharm_Projects/Sentiment_Analysis/models/saga_logistic_regression.model'

if fit:
    saga_lr = LogisticRegression(solver='saga', max_iter=int(10e4), n_jobs=-1)
    saga_lr.fit(x_train, y_train)
    joblib.dump(saga_lr, saga_lr_file)
else:
    if not os.path.isfile(saga_lr_file):
        saga_lr = LogisticRegression(solver='saga', max_iter=int(10e4), n_jobs=-1)
        saga_lr.fit(x_train, y_train)
        joblib.dump(saga_lr, saga_lr_file)
    else:
        saga_lr = joblib.load(saga_lr_file)

# %%
print("Accuracy of the model is", saga_lr.score(x_test, y_test) * 100, "% - Logistic Regression")
# %%
text = ""
test_data = vec.transform(
    [(preprocess(text, punc, contractions.fix, lemm.lemmatize, word_tokenize))])
y_pred = saga_lr.predict(x_test)

# %%
cm = confusion_matrix(y_test, y_pred)
cm_plot = sns.heatmap(cm, annot=True, cmap='Reds', fmt='d')
cm_plot.set_xlabel('Predicted Values')
cm_plot.set_ylabel('Actual Values')
plt.show()

_TN = cm[1][1]
_FP = cm[1][0]
_FN = cm[0][1]
_TP = cm[0][0]

print(f'TP: {_TP}')
print(f'FN: {_FN}')
print(f'FP: {_FP}')
print(f'TN: {_TN}')

print("F1", f1_score(y_test, y_pred))
print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("Matthew's Coefficient is", matthews_corrcoef(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
