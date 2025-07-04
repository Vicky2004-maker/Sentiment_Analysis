import contractions
import numpy as np
import pandas as pd
import string
from nltk import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from sklearn.model_selection import cross_val_score
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
xgb_file = 'D:/PyCharm_Projects/Sentiment_Analysis/models/xgb_classifier.model'

if fit:
    def objective(params):
        clf = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            eval_metric='logloss',
            n_jobs=-1,
            random_state=1,
        )
        score = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=3).mean()
        return {'loss': -score, 'status': STATUS_OK}

    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials,
        rstate=np.random.default_rng(1),
    )
    best['n_estimators'] = int(best['n_estimators'])
    best['max_depth'] = int(best['max_depth'])
    xgb_clf = XGBClassifier(
        n_estimators=best['n_estimators'],
        max_depth=best['max_depth'],
        learning_rate=best['learning_rate'],
        subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'],
        eval_metric='logloss',
        n_jobs=-1,
        random_state=1,
    )
    xgb_clf.fit(x_train, y_train)
    joblib.dump(xgb_clf, xgb_file)
else:
    if not os.path.isfile(xgb_file):
        xgb_clf = XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=1)
        xgb_clf.fit(x_train, y_train)
        joblib.dump(xgb_clf, xgb_file)
    else:
        xgb_clf = joblib.load(xgb_file)

# %%
print("Accuracy of the model is", xgb_clf.score(x_test, y_test) * 100, "% - XGBoost")
# %%
text = ""
test_data = vec.transform(
    [(preprocess(text, punc, contractions.fix, lemm.lemmatize, word_tokenize))])
y_pred = xgb_clf.predict(x_test)

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
