import streamlit as st
import pickle
import numpy as np

def load_model():
    with open("saved_steps.pkl","rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_education = data["le_education"]
le_country = data["le_country"]

def show_predict_page():
    st.title("Software Developer Salary Prediction 2023")

    st.write("""
    ### Select the following information to predict the salary for the year 2023
    """)

    countries = (
        "United States of America",
        "Germany",
        "India",
        "United Kingdom of Great Britain and Northern Ireland",
        "Canada",
        "France",
        "Netherlands",
        "Australia",
        "Brazil",
        "Spain",
        "Sweden",
        "Italy",
        "Poland",
        "Switzerland",
        "Denmark",
        "Norway",
        "Israel"

    )

    education = (
        "Bachelor’s degree",
        "Less than a Bachelors",
        "Master’s degree",
        "Post grad"
    )

    country = st.selectbox("Country",countries)
    education = st.selectbox("Education",education)


    experience = st.slider("Years of experience",0,50,3)

    ok = st.button("Calculate Salary")

    if ok:
        X = np.array([[country,education,experience]])
        X[:,0] = le_country.transform(X[:,0])
        X[:,1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader("The estimated salary is ${:,.02f}".format(salary[0]))
