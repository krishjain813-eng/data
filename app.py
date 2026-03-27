
import streamlit as st
import pandas as pd
import pickle

clf = pickle.load(open("clf.pkl","rb"))
reg = pickle.load(open("reg.pkl","rb"))

st.title("Fintech Decision Engine")

uploaded = st.file_uploader("Upload CSV")

if uploaded:
    df = pd.read_csv(uploaded)
    pred = clf.predict(df)
    prob = clf.predict_proba(df)[:,1]
    loan = reg.predict(df)

    df["interest_prediction"] = pred
    df["probability"] = prob
    df["recommended_loan"] = loan

    st.write(df)
