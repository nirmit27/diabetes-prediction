import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler as SS


# Data fetching, preprocessing and model prediction

ss = SS()
data = []
features = ["Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

model = pickle.load(open("trained_model.pkl", 'rb'))


def prediction(*data):
    data = np.asarray(data)
    data = data.reshape(-1, 1)
    data = ss.fit_transform(data)

    res = model.predict(data.reshape(1, -1))[0]
    return res


# Web-app configuration

def main():
    st.set_page_config(page_title="Diabetes Prediction",
                       page_icon="⚕️", layout="centered")

    st.markdown('''
        <h2 align="center" style="color: lightblue; font-weight: bolder; margin-top: -60px;">Diabetes Prediction</h2><br>''', unsafe_allow_html=True)

    with st.form("patient-data", border=True):
        st.subheader("Patient Data")

        c1, c2 = st.columns((0.5, 0.5), gap="medium")

        with c1:
            pregnancies = st.number_input("Pregnancy count", min_value=0)
            glucose = st.number_input("Glucose level", min_value=0)
            blood_pressure = st.number_input("Blood pressure", min_value=0)
            skin_thickness = st.number_input(
                "Skin thickness", min_value=0)

        with c2:
            insulin = st.number_input(
                "Insulin level", min_value=0)
            bmi = st.number_input("BMI", format="%.1f")
            dpf = st.number_input(
                "Diabetes Pedigree Function", format="%.3f")
            age = st.number_input("Age", min_value=0)

        submitted = st.form_submit_button("Submit")

    result_placeholder = st.empty()

    if submitted:
        data.extend([pregnancies, glucose, blood_pressure,
                     skin_thickness, insulin, bmi, dpf, age])
        result = prediction(data)
        result_placeholder.markdown(
            f'''<h4 align="center" style="color: {"lightred" if result else "skyblue"}; margin-block: 15px; border: solid {"#e50914" if result else "#3f79d7"} 1px; border-radius: 10px; padding: 20px; margin-inline: auto; text-align: center;">{"Diabetes suspected." if result else "Diabetes not suspected."}</h4>''', unsafe_allow_html=True)
    else:
        result_placeholder.markdown(
            '''<h4 style="color: grey; margin-block: 15px; border: solid #1b3053 1px; border-radius: 10px; padding: 20px; margin-inline: auto; text-align: center;">Submit for results</h4>''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
