import requests
import streamlit as st

st.title("Cat or Dog?")

file = st.file_uploader("Choose file to predict", type=["jpg", "png"])

if file is not None:
    st.toast("File uploaded, detecting...")

    try:
        response = requests.post("http://api:8000/predict", files={"file": file.getvalue()})

        if response.status_code != 200:
            raise Exception()

        st.write(f"It's a {response.json()}!")

    except Exception:
        st.error("Cannot detect!")
