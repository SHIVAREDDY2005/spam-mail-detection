import  streamlit as  st
import pickle
import nltk
from nltk.corpus import stopwords
stop=stopwords.words('english')

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def preprocess(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    res=[]
    for i in text:
        if(i.isalnum()):
            if(i not  in stop):
                res.append(ps.stem(i))
    return " ".join(res)
    
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email spam detection")
input=st.text_input('Enter your meassge.....')
if st.button('check'):
    preprocessed=preprocess(input)
    vector_input=tfidf.transform([preprocessed])
    res=model.predict(vector_input)[0]
    if  res==1:
        st.header('spam')
    else:
        st.header("not spam")