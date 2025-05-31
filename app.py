import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load data and train model
@st.cache_resource
def train_model():
    data = pd.read_csv(r"C:\Users\rakesh\Desktop\END TO END FAKE NEWS DETECTION\fake_or_real_news.csv")
    x = np.array(data["title"])
    y = np.array(data["label"])
    
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(xtrain, ytrain)
    return model, cv

# Load model and vectorizer
model, cv = train_model()

# Streamlit UI
st.title("📰 Fake News Detection")
st.write("Enter a news headline to predict whether it is **FAKE** or **REAL**.")

user_input = st.text_input("Enter News Title:")

if st.button("Predict"):
    if user_input:
        input_data = cv.transform([user_input])
        prediction = model.predict(input_data)
        st.success(f"The news is predicted to be: **{prediction[0].upper()}**")
    else:
        st.warning("Please enter a news title to predict.")
