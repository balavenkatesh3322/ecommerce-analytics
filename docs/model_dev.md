## Machine Learning Model for Actionable Logistics Insights

This section details the development and implementation of a machine learning model designed to predict delivery delays for the Olist e-commerce platform. The model is a key component of the overall analytics solution, providing actionable, forward-looking insights for the logistics team.

### 1. Objective: Proactive Delay Prediction
The primary goal of the machine learning model is to answer a critical business question for the logistics team:

"Can we predict, at the time of purchase, if an order is at a high risk of being delivered late?"

By creating a reliable predictive model, the logistics team can move from a reactive to a proactive stance. Instead of only analyzing why past deliveries were late, they can identify at-risk orders in real-time and take preemptive action, such as:
- Flagging the order for enhanced monitoring.
- Prioritizing its fulfillment and shipping.
- Exploring alternative, faster shipping options.

### 2. Model Features (The "Inputs")
Based on the available data, we selected a set of features that are available at or shortly after the time of purchase and are likely to influence delivery time.

The key features used to train the model are:

**Geographical Data**:
- `seller_state`: The state where the seller is located.
- `customer_state`: The state where the customer is located.

**Product Information**:
- `product_category_name_english`: The category of the product being shipped.
- `product_weight_g`: The weight of the product in grams.

**Order Economics**:
- `price`: The price of the product.
- `freight_value`: The shipping cost paid by the customer.

### 3. Model Selection: LightGBM
We chose a LightGBM (Light Gradient Boosting Machine) classifier for this task. LightGBM is a powerful and widely-used gradient boosting framework for the following reasons:
- **High Performance**: It is known for its high accuracy and efficiency on tabular datasets like this one.
- **Fast Training Speed**: It is significantly faster than other gradient boosting models, making it ideal for rapid iteration and for use in an interactive application.
- **Handles Categorical Features**: While we use a One-Hot Encoder for maximum compatibility in our pipeline, LightGBM has built-in capabilities to handle categorical data efficiently.

### 4. Preprocessing and Pipeline
To ensure the model receives clean, standardized data, we use a scikit-learn Pipeline. This bundles preprocessing steps and the model into a single object, which prevents data leakage and makes the model easy to deploy.

The pipeline performs two main steps:
- **One-Hot Encoding**: Categorical features like `seller_state` and `product_category_name_english` are converted into a numerical format that the model can understand.
- **Standard Scaling**: Numerical features like `price` and `product_weight_g` are scaled to have a mean of 0 and a standard deviation of 1. This helps the model converge faster and prevents features with large values from dominating the learning process.

### 5. Model Evaluation (Measuring Success)
After training the model on a portion of the data, we evaluate its performance on a separate, unseen test set. The key metrics we use are:
- **Accuracy**: The overall percentage of correct predictions.
- **Classification Report**: Provides precision, recall, and F1-score for both "On-Time" and "Late" classes. This is crucial because it tells us how well the model identifies each class, not just its overall accuracy.
- **ROC AUC Score**: A measure of the model's ability to distinguish between the two classes. A score closer to 1.0 indicates a better model.
- **Confusion Matrix**: A visual breakdown of the model's predictions, showing:
  - True Positives (Late orders correctly identified)
  - True Negatives (On-time orders correctly identified)
  - False Positives (On-time orders incorrectly flagged as late)
  - False Negatives (Late orders that the model missed)

These metrics are all calculated and displayed within the "Model Training & Evaluation" page of the provided Streamlit application, giving the user a comprehensive view of the model's effectiveness.
