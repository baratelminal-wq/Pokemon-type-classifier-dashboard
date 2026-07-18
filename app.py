from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests # For linking to data source

# --- Configuration ---
PROJECT_DIR = Path(__file__).resolve().parent


def resolve_file(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


st.set_page_config(page_title="Pokémon Type Classifier Dashboard", layout="wide")

# --- Caching Data and Resources ---
@st.cache_data
def load_data():
    raw_path = resolve_file(PROJECT_DIR / "data" / "raw_data.csv", PROJECT_DIR / "raw_data.csv")
    clean_path = resolve_file(PROJECT_DIR / "data" / "clean_data.csv", PROJECT_DIR / "clean_data.csv")

    df_clean = pd.read_csv(clean_path)
    if raw_path == clean_path:
        df_raw = df_clean.copy()
    else:
        df_raw = pd.read_csv(raw_path)
    return df_raw, df_clean

@st.cache_resource
def load_model_artifacts():
    model_path = resolve_file(PROJECT_DIR / "model" / "model.pkl", PROJECT_DIR / "model.pkl")
    scaler_path = resolve_file(PROJECT_DIR / "model" / "scaler.pkl", PROJECT_DIR / "scaler.pkl")
    label_encoder_path = resolve_file(PROJECT_DIR / "model" / "label_encoder.pkl", PROJECT_DIR / "label_encoder.pkl")
    metrics_path = resolve_file(PROJECT_DIR / "model" / "model_metrics.json", PROJECT_DIR / "model_metrics.json")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    with open(metrics_path, "r") as f:
        model_metrics = json.load(f)
    return model, scaler, label_encoder, model_metrics

# Load all assets
df_raw, df_clean = load_data()
model, scaler, le, model_metrics = load_model_artifacts()

FEATURES = model_metrics['features']
CLASS_LABELS = model_metrics['class_labels']

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. Project Overview",
    "2. Data Overview",
    "3. Exploratory Data Analysis",
    "4. Model Performance",
    "5. Live Prediction"
])

# --- Page Content ---

# 1. Project Overview
if page == "1. Project Overview":
    st.title("Pokémon Type Classification Dashboard")
    st.write(
        "This dashboard presents an end-to-end data science project, "
        "from data acquisition and cleaning to machine learning model deployment. "
        "The goal is to predict a Pokémon's primary type based on its base stats and other characteristics."
    )
    st.markdown("**Data Source:** [PokéAPI](https://pokeapi.co/)")

    st.header("Key Project Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pokémon", f"{len(df_clean)} records")
    with col2:
        st.metric("Number of Types", f"{len(CLASS_LABELS)} classes")
    with col3:
        best_model_name = model_metrics['best_model'].replace('_', ' ').title()
        st.metric("Best Model (Accuracy)", f"{best_model_name}: {model_metrics['all_model_results'][0]['accuracy']:.2f}")

# 2. Data Overview
elif page == "2. Data Overview":
    st.title("Data Overview")
    st.write("A look at the raw and cleaned datasets.")

    tab1, tab2 = st.tabs(["Raw Data", "Cleaned Data"])
    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head())
        st.write(f"Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
        st.subheader("Missing Values (Raw Data)")
        st.dataframe(df_raw.isna().sum().to_frame(name='Missing Count'))

    with tab2:
        st.subheader("Cleaned Data Preview")
        st.dataframe(df_clean.head())
        st.write(f"Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        st.subheader("Missing Values (Cleaned Data)")
        st.dataframe(df_clean.isna().sum().to_frame(name='Missing Count'))

# 3. Exploratory Data Analysis
elif page == "3. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.write("Interactive charts to explore Pokémon data.")

    st.subheader("Class Balance of Primary Types")
    type_counts = df_clean['primary_type'].value_counts().reset_index()
    type_counts.columns = ['Primary Type', 'Count']
    fig_type_balance = px.bar(
        type_counts, x='Primary Type', y='Count',
        title='Distribution of Primary Pokémon Types',
        color='Primary Type',
        template='plotly_white'
    )
    st.plotly_chart(fig_type_balance, use_container_width=True)

    st.subheader("Correlation Heatmap of Numeric Features")
    numeric_df = df_clean[FEATURES]
    corr_matrix = numeric_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap of Numeric Pokémon Features"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Stat Distribution by Primary Type")
    selected_stat = st.selectbox("Select a Stat for Box Plot", FEATURES[3:]) # Exclude height, weight, base_experience from dropdown as we have many stats
    if selected_stat:
        fig_boxplot = px.box(
            df_clean, x='primary_type', y=selected_stat,
            title=f'{selected_stat.replace("_", " ").title()} Distribution by Primary Pokémon Type',
            color='primary_type',
            template='plotly_white'
        )
        st.plotly_chart(fig_boxplot, use_container_width=True)

# 4. Model Performance
elif page == "4. Model Performance":
    st.title("Model Performance")
    st.write("Evaluation metrics and confusion matrix for the trained models.")

    st.subheader("Model Comparison")
    results_df = pd.DataFrame(model_metrics['all_model_results'])
    results_df_styled = results_df.set_index('model').style.format("{:.3f}")
    st.dataframe(results_df_styled, use_container_width=True)

    st.subheader(f"Confusion Matrix for {model_metrics['best_model'].replace('_', ' ').title()}")
    cm_df = pd.DataFrame(
        model_metrics['confusion_matrix'],
        index=[f'Actual {c}' for c in CLASS_LABELS],
        columns=[f'Predicted {c}' for c in CLASS_LABELS]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale='Blues',
        title=f"Confusion Matrix for {model_metrics['best_model'].replace('_', ' ').title()}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Feature Importance")
    feature_importance_df = pd.DataFrame({
        'Feature': list(model_metrics['feature_importance'].keys()),
        'Importance': list(model_metrics['feature_importance'].values())
    }).sort_values('Importance', ascending=False)

    fig_fi = px.bar(
        feature_importance_df, x='Importance', y='Feature',
        orientation='h', title='Feature Importance for Best Model',
        template='plotly_white'
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# 5. Live Prediction
elif page == "5. Live Prediction":
    st.title("Live Prediction")
    st.write("Input Pokémon stats to predict its primary type.")

    input_data = {}
    with st.form("prediction_form"):
        st.subheader("Input Pokémon Stats:")
        col1, col2, col3 = st.columns(3)

        # Dynamically create input widgets for each feature
        for i, feature in enumerate(FEATURES):
            if i % 3 == 0: col = col1
            elif i % 3 == 1: col = col2
            else: col = col3

            with col:
                min_val = df_clean[feature].min()
                max_val = df_clean[feature].max()
                avg_val = df_clean[feature].mean()
                input_data[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(avg_val)
                )

        submit_button = st.form_submit_button(label="Predict Pokémon Type")

    if submit_button:
        # Convert input to DataFrame for scaling
        input_df = pd.DataFrame([input_data])

        # Scale the input features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction_encoded = model.predict(input_scaled)
        predicted_type = le.inverse_transform(prediction_encoded)[0]

        st.subheader("Prediction Result:")
        st.success(f"The predicted primary type is: **{predicted_type.upper()}**")

        # Display prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            proba_df = pd.DataFrame({
                'Type': CLASS_LABELS,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)

            fig_proba = px.bar(
                proba_df, x='Probability', y='Type',
                orientation='h', title='Prediction Probabilities',
                color='Type',
                template='plotly_white'
            )
            st.plotly_chart(fig_proba, use_container_width=True)

# --- Footer ---
st.sidebar.markdown("""
---
**Data Source:** PokéAPI
**Last Updated:** June 2024
""")