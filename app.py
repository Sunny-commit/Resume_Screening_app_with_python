import numpy as np
import re
import pickle
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
victorizer=TfidfVectorizer(stop_words='english')
nltk.download('punkt')
nltk.download('stopwords')

#loading pickle modles
clf=pickle.load(open('clf.pkl','rb'))
victorizer=pickle.load(open('tfidf.pkl','rb'))
import re


def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters, punctuations, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


#web app
def main():
    st.title('Resume Screening App')
    uploaded_file=st.file_uploader('Upload Resume',type=['txt','pdf'])
    if uploaded_file is not None:
        try:
            resume_bytes=uploaded_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')
        cleaned_resume=clean_text(resume_text)
        cleaned_resume=victorizer.transform([cleaned_resume])
        prediction_id=clf.predict(cleaned_resume)[0]
        st.write(prediction_id)






if __name__=='__main__':
    main()