import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


docs = pd.read_csv('nlp_data.csv')
eval = docs[1020:]
docs = docs.dropna()
X = docs.lineitem
y = docs.relevance
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = CountVectorizer(stop_words='english')
vect.fit(X_train)
# print(vect.vocabulary_)
X_train_transformed = vect.transform(X_train)
X_test_transformed =vect.transform(X_test)
mnb = MultinomialNB()
mnb.fit(X_train_transformed,y_train)
y_pred_class = mnb.predict(X_test_transformed)
y_pred_proba =mnb.predict_proba(X_test_transformed)

print(metrics.accuracy_score(y_test, y_pred_class))

joblib.dump(mnb, 'NB_model.pkl')
joblib.dump(vect, 'NB_vect.pkl')

# eval_data = eval.lineitem
# eval_data.dropna(inplace=True)
# print(eval_data)
# eval_transformed  = vect.transform(eval_data)
# eval_class = mnb.predict(eval_transformed)
# df = pd.DataFrame({"lineitem":eval_data,"relevance":eval_class})
# df.to_csv("checkEval.csv")
