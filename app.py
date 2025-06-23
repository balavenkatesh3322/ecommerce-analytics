import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import functions from our src folder
from src.data_processing import load_and_prepare_data
from src.model_training import train_and_evaluate_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Olist End-to-End Logistics Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- App State Management ---
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

# --- Data Loading Logic ---
# This block runs only once at the start or if data isn't loaded yet.
if not st.session_state.data_loaded:
    df = load_and_prepare_data(base_path='dataset')
    if df is not None:
        st.session_state['df'] = df
        st.session_state['data_loaded'] = True
    else:
        # This part handles the case where the folder is missing on first load
        st.session_state['df'] = None
        st.session_state['data_loaded'] = False


# --- Sidebar ---
st.sidebar.title("Olist Logistics Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Exploratory Data Analysis", "⚙️ Model Training & Evaluation", "🔮 Delivery Delay Predictor"])
st.sidebar.markdown("---")

# --- Page Content ---

if page == "🏠 Home":
    st.title("Welcome to the Olist E-commerce Logistics Dashboard")
    st.markdown("""
    This interactive dashboard provides a complete, end-to-end solution for analyzing and predicting logistics performance on the Olist platform. The code is structured with `src` and `dataset` folders for modularity and clarity.

    **Navigate using the sidebar to:**
    1.  **📊 Exploratory Data Analysis:** Understand key trends in deliveries, delays, and product categories.
    2.  **⚙️ Model Training & Evaluation:** Train a machine learning model to predict delivery delays and see how well it performs.
    3.  **🔮 Delivery Delay Predictor:** Use the trained model to get live predictions for new orders.
    """)
    st.markdown("---")
    st.header("Current Status")
    if st.session_state.get('data_loaded', False) and st.session_state.df is not None:
        st.success("✅ Data successfully loaded from the `dataset` folder.")
        st.write("Prepared Data Sample:")
        st.dataframe(st.session_state['df'].head())
    else:
        st.error("Data not loaded. Please ensure the `dataset` folder exists at the root of the project and contains all required Olist CSV files.")
        st.info("After placing the files in the `dataset` folder, please refresh the page.")


elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    if not st.session_state.get('data_loaded', False):
        st.warning("Data not loaded. Please go to the Home page and ensure your `dataset` folder is set up correctly.")
    else:
        df = st.session_state['df']
        st.markdown("### Overall Delivery Status")
        
        col1, col2 = st.columns(2)
        status_counts = df['is_late'].value_counts()
        status_counts.index = ['On-Time', 'Late']
        col1.metric("Total Delivered Orders", f"{df.shape[0]:,}")
        col2.metric("On-Time Delivery Rate", f"{1 - (status_counts.get('Late', 0) / df.shape[0]):.2%}")

        fig_status, ax_status = plt.subplots()
        sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax_status, palette=["#2ecc71", "#e74c3c"])
        ax_status.set_title('On-Time vs. Late Deliveries')
        ax_status.set_ylabel('Number of Orders')
        st.pyplot(fig_status)

        st.markdown("### Top 10 Product Categories with Most Delays")
        late_df = df[df['is_late'] == 1]
        top_late_cats = late_df['product_category_name_english'].value_counts().nlargest(10)
        fig_cats, ax_cats = plt.subplots(figsize=(10, 6))
        sns.barplot(y=top_late_cats.index, x=top_late_cats.values, orient='h', ax=ax_cats, palette='viridis')
        ax_cats.set_title('Top 10 Product Categories with Most Delays')
        ax_cats.set_xlabel('Number of Late Orders')
        st.pyplot(fig_cats)


elif page == "⚙️ Model Training & Evaluation":
    st.title("⚙️ Model Training & Evaluation")
    if not st.session_state.get('data_loaded', False):
        st.warning("Data not loaded. Please go to the Home page and ensure your `dataset` folder is set up correctly.")
    else:
        if st.button("🚀 Train Model Now", key="train_button"):
            with st.spinner("Training the LightGBM model... This may take a moment."):
                df = st.session_state['df']
                pipeline, results = train_and_evaluate_model(df)
                st.session_state['model_pipeline'] = pipeline
                st.session_state['evaluation_results'] = results
                st.session_state['model_trained'] = True

        if st.session_state.get('model_trained', False):
            st.success("✅ Model training and evaluation complete!")
            results = st.session_state['evaluation_results']

            col1, col2 = st.columns(2)
            col1.metric("Model Accuracy", f"{results['accuracy']:.4f}")
            col2.metric("ROC AUC Score", f"{results['roc_auc']:.4f}")

            st.markdown("### Classification Report")
            st.dataframe(pd.DataFrame(results['report']).transpose())

            fig1, ax1 = plt.subplots()
            sns.heatmap(results['cm'], annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=['Predicted On-Time', 'Predicted Late'],
                        yticklabels=['Actual On-Time', 'Actual Late'])
            ax1.set_title('Confusion Matrix')
            
            fpr, tpr = results['roc_curve']
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {results['roc_auc']:.2f})")
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax2.legend(loc="lower right")

            col1_viz, col2_viz = st.columns(2)
            with col1_viz: st.pyplot(fig1)
            with col2_viz: st.pyplot(fig2)


elif page == "🔮 Delivery Delay Predictor":
    st.title("🔮 Delivery Delay Predictor")
    if not os.path.exists('delivery_model.joblib'):
        st.warning("Model not found. Please train a model first on the 'Model Training & Evaluation' page.")
    elif not st.session_state.get('data_loaded', False):
        st.warning("Data not loaded. Please go to the Home page and ensure your `dataset` folder is set up correctly.")
    else:
        model = joblib.load('delivery_model.joblib')
        df = st.session_state['df']

        seller_states = sorted(df['seller_state'].unique())
        customer_states = sorted(df['customer_state'].unique())
        product_categories = sorted(df['product_category_name_english'].unique())

        st.sidebar.header("Enter Order Details")
        seller_state = st.sidebar.selectbox("Seller State", options=seller_states, index=seller_states.index('SP'))
        customer_state = st.sidebar.selectbox("Customer State", options=customer_states, index=customer_states.index('SP'))
        product_category = st.sidebar.selectbox("Product Category", options=product_categories, index=product_categories.index('bed_bath_table'))
        price = st.sidebar.number_input("Product Price (R$)", min_value=0.0, value=100.0, step=10.0)
        freight_value = st.sidebar.number_input("Freight Value (R$)", min_value=0.0, value=20.0, step=5.0)
        product_weight_g = st.sidebar.slider("Product Weight (grams)", min_value=int(df['product_weight_g'].min()), max_value=40000, value=1000, step=50)

        if st.sidebar.button("Predict Delivery Status", type="primary"):
            input_df = pd.DataFrame({
                'seller_state': [seller_state], 'customer_state': [customer_state],
                'product_category_name_english': [product_category], 'price': [price],
                'freight_value': [freight_value], 'product_weight_g': [product_weight_g]
            })
            
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error("Prediction: High Risk of Delay", icon="�")
                st.metric(label="Probability of Delay", value=f"{prediction_proba[1]:.2%}")
                st.progress(prediction_proba[1])
            else:
                st.success("Prediction: Likely On-Time", icon="✅")
                st.metric(label="Probability of Being On-Time", value=f"{prediction_proba[0]:.2%}")
                st.progress(prediction_proba[0])

            with st.expander("Show Input Data"): st.dataframe(input_df)