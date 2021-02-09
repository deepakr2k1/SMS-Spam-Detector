import numpy as np
import pandas as pd
import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

model = pickle.load(open('model.pkl', 'rb'))
st.title("SMS Spam Detector")

msg = st.text_input("Enter the Message to test")

_columns = pd.read_csv('columns.csv')

def preprocess_for_new_input(sents):
    new_input_x = pd.DataFrame(np.zeros((1,6000)),columns = _columns.columns)
    sents = re.sub('[^a-zA-Z]',' ', sents)
    sents = sents.lower()
    _words = nltk.word_tokenize(sents)
    _words = [lemmatizer.lemmatize(_word) for _word in _words if _word not in set(stopwords.words('english'))]
    for i in range(len(_words)):
        if _words[i] in _columns.columns:
            new_input_x[_words[i]]=1;
    return new_input_x;

msg_x = preprocess_for_new_input(msg)
ans = model.predict(msg_x)

if(msg):
    if(ans[0]==0):
        st.write("""
        ### It is a Ham
        """)
    elif(ans[0]==1):
        st.write("""
        ### It is a Spam
        """)