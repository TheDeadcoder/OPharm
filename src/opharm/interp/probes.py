import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = (1e-3, 1e-2, 1e-1, 1.0)


def _model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def fit_probe(x, y, groups, cs=CS, folds=5, splits=None):
    splits = splits or list(GroupKFold(min(folds, len(set(groups)))).split(x, y, groups))
    cv = {}
    for c in cs:
        scores = []
        for tr, va in splits:
            m = _model(c).fit(x[tr], y[tr])
            scores.append(roc_auc_score(y[va], m.decision_function(x[va])))
        cv[c] = float(np.mean(scores))
    best = max(cv, key=cv.get)
    return _model(best).fit(x, y), cv[best], best


def ngram_scores(texts_train, y_train, texts_test):
    out = {}
    for name, clf in (("naive_bayes", MultinomialNB()), ("logistic", LogisticRegression(max_iter=3000))):
        pipe = make_pipeline(CountVectorizer(ngram_range=(1, 3), min_df=2), clf).fit(texts_train, y_train)
        out[name] = pipe.predict_proba(texts_test)[:, 1]
    return out


def ngram_auroc(texts_train, y_train, texts_test, y_test):
    return {k: float(roc_auc_score(y_test, v)) for k, v in ngram_scores(texts_train, y_train, texts_test).items()}
