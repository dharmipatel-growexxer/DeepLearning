import streamlit as st
import pandas as pd

from src.predict import (
    PreprocessConfig,
    train_for_inference,
    predict_file,
)

st.set_page_config(page_title="Readmission Predictor", layout="wide")

st.title("City General Hospital — 30-day Readmission Predictor")
st.write(
    "Upload a CSV (same columns as test.csv). The app will train the model on train.csv and output predictions."
)

cfg = PreprocessConfig()

@st.cache_resource
def load_model():
    return train_for_inference("data/train.csv", cfg)

model_bundle = load_model()

uploaded = st.file_uploader("Upload input CSV", type=["csv"])

if uploaded is not None:
    input_df = pd.read_csv(uploaded)
    temp_in = "/tmp/streamlit_input.csv"
    temp_out = "/tmp/streamlit_predictions.csv"
    input_df.to_csv(temp_in, index=False)

    if isinstance(model_bundle, tuple):
        preprocessor, model = model_bundle
        # Use predict_file to keep behavior consistent
        predict_file(preprocessor, model, temp_in, temp_out, cfg)
    else:
        # Backward compatibility if model_bundle is a pipeline
        predict_file(model_bundle, temp_in, temp_out, cfg)

    preds = pd.read_csv(temp_out)
    st.subheader("Predictions")
    st.dataframe(preds, use_container_width=True)

    st.download_button(
        "Download predictions.csv",
        data=preds.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV file to generate predictions.")
