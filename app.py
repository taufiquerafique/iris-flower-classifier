import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/iris_model.pkl")

@st.cache_resource
def load_data():
    return pd.read_csv("data/iris.csv")

accuracy = joblib.load(
    "models/accuracy.pkl"
)

model = load_model()
df = load_data()

st.title("🌸 Iris Flower Classifier")

st.write(
    """
    Predict Iris flower species using a trained
    Decision Tree Machine Learning model.
    """
)

with st.sidebar:
    st.header("About Project")
    st.write("""
    This project predicts Iris flower species
    using a Decision Tree Classifier.
    """)
    st.markdown("---")
    st.write("Developer")
    st.write("Taufique Rafique")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Flower Measurements")

    sepal_length = st.number_input(
        "Sepal Length",
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width",
        value=3.5
    )

    petal_length = st.number_input(
        "Petal Length",
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width",
        value=0.2
    )

    predict = st.button("Predict Flower")

with col2:
    st.subheader("Prediction")

    if predict:
        prediction = model.predict(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]
        )
            
        flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
        ]

        st.success(
            flower_names[prediction[0]]
        )

        probability = model.predict_proba(
            [
                [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
                ]
            ]
        )

        st.write("Prediction Confidence")

        st.progress(float(probability.max()))
        st.write(f"{probability.max()*100:.2f}%")

        st.markdown("---")

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.write("Rows: ", df.shape[0], "Columns: ", df.shape[1])

        st.metric(
        label="Model Accuracy",
        value=f"{accuracy*100:.2f}%"
        )

        