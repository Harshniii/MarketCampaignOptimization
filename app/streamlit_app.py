import streamlit as st
import pandas as pd
from textblob import TextBlob
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = r'C:\Users\Aareh\MarketCampaignOptimization\models\marketing_model.pkl'

# Load the model
model = joblib.load(MODEL_PATH)

# App Title
st.set_page_config(page_title="Marketing Campaign Optimizer", layout="wide")
st.title("📊 Marketing Campaign Optimization")
st.markdown("Leverage sentiment analysis & ML to refine marketing strategies.")

# Sidebar Navigation
option = st.sidebar.radio("Select Feature", ["📢 Sentiment Analyzer", "📁 Campaign Data Predictor"])

# Sentiment Analyzer
if option == "📢 Sentiment Analyzer":
    st.subheader("Analyze Customer Feedback or Campaign Content")
    text_input = st.text_area("Enter campaign message:")

    if st.button("Analyze Sentiment"):
        if text_input:
            analysis = TextBlob(text_input)
            sentiment_score = analysis.sentiment.polarity

            st.write(f"**Sentiment Score:** `{sentiment_score}`")
            if sentiment_score > 0:
                st.success("Positive Sentiment 😊")
            elif sentiment_score < 0:
                st.warning("Negative Sentiment 😟")
            else:
                st.info("Neutral Sentiment 😐")
        else:
            st.error("Please enter some text for analysis.")

# Campaign Data Predictor
elif option == "📁 Campaign Data Predictor":
    st.subheader("Upload Campaign Data for Predictions")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Read the uploaded CSV
        data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", data.head())

        # Drop columns that were not used during model training
        columns_to_drop = ['Campaign_ID', 'Acquisition_Cost']
        data = data.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' handles if columns are missing

        # One-hot encoding for categorical variables (apply the same encoding as during training)
        data = pd.get_dummies(data, columns=['Campaign_Type', 'Channel_Used'], drop_first=True)

        # Ensure the same columns as in the model training set
        model_columns = model.feature_names_in_  # For models like RandomForest, XGBoost, etc.
        missing_cols = set(model_columns) - set(data.columns)
        extra_cols = set(data.columns) - set(model_columns)
        
        # Handle missing columns (add them with default values)
        for col in missing_cols:
            data[col] = 0  # or use np.nan depending on your model's handling of missing values

        # Align columns to the model's expected order
        data = data[model_columns]  # This ensures the column order matches the training data

        # Make predictions
        try:
            predictions = model.predict(data)
            data["Predicted Outcome"] = predictions
            st.success("✅ Predictions completed.")
            st.write(data)

            st.write("### 📈 Campaign Result Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x="Predicted Outcome", data=data, ax=ax, palette="Set2")
            ax.set_title("Predicted Outcome Distribution")
            st.pyplot(fig)

            st.write("### 📊 Feature Correlation Heatmap")
            corr = data.drop(columns=["Predicted Outcome"]).corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, vmin=-1, vmax=1)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)

            st.write("### 🎯 Engagement Score vs Sentiment Polarity")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=data,
                x="Sentiment_Polarity",
                y="Engagement_Score",
                hue="Predicted Outcome",
                palette="coolwarm",
                ax=ax
            )
            ax.set_title("Engagement Score vs Sentiment Polarity")
            st.pyplot(fig)

            st.write("### 🥧 Pie Chart: Predicted Outcome Distribution")

            # Count occurrences of each prediction
            outcome_counts = data["Predicted Outcome"].value_counts()

            fig, ax = plt.subplots()
            colors = sns.color_palette('pastel')[0:len(outcome_counts)]

            ax.pie(
                outcome_counts,
                labels=outcome_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                wedgeprops={'edgecolor': 'black'}
            )
            ax.set_title("Predicted Outcome Distribution")
            st.pyplot(fig)


        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
            st.write(f"Error details: {e}")