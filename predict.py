from sklearn.externals import joblib
# from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
NB_model = open('NB_model.pkl','rb')
mnb = joblib.load(NB_model)
NB_vect = open('NB_vect.pkl','rb')
vect = joblib.load(NB_vect)
sent = pd.Series(["current asset"])

sent_transformed = vect.transform(sent)
ans = mnb.predict(sent_transformed)[0]
print(int(ans))
