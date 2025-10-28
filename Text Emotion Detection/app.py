import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
# Define the model path relative to this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "text_emotion.pkl")

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at {MODEL_PATH}. Please make sure 'text_emotion.pkl' is inside the 'model' folder.")
    st.stop()

# Load the trained model
pipe_lr = joblib.load(MODEL_PATH)

# Mapping of emotions to emojis
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# Function to predict emotion
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Streamlit app
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions in Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text and raw_text.strip() != "":
        col1, col2 = st.columns(2)

        # Get prediction and probability
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        # Display original text and prediction
        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        # Display probability chart
        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            )
            st.altair_chart(fig, use_container_width=True)
    elif submit_text:
        st.warning("Please type some text before submitting.")

if __name__ == '__main__':
    main()
