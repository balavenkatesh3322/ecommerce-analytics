import pandas as pd
import streamlit as st
import os


@st.cache_data
def load_and_prepare_data(base_path="dataset"):
    """
    Loads all required datasets from the specified path, performs joins,
    and conducts feature engineering.

    Args:
        base_path (str): The path to the folder containing the CSV files.

    Returns:
        pandas.DataFrame: A clean, prepared dataframe for analysis and modeling.
    """
    try:
        # Define file paths
        files = {
            "orders": os.path.join(base_path, "olist_orders_dataset.csv"),
            "items": os.path.join(base_path, "olist_order_items_dataset.csv"),
            "products": os.path.join(base_path, "olist_products_dataset.csv"),
            "customers": os.path.join(base_path, "olist_customers_dataset.csv"),
            "sellers": os.path.join(base_path, "olist_sellers_dataset.csv"),
            "category_translation": os.path.join(
                base_path, "product_category_name_translation.csv"
            ),
        }

        # Load dataframes
        data = {name: pd.read_csv(path) for name, path in files.items()}

    except FileNotFoundError as e:
        st.error(
            f"Error loading data: {e}. Make sure the `dataset` folder is correctly placed and populated."
        )
        return None

    # Merge datasets
    df = data["orders"].merge(data["items"], on="order_id", how="left")
    df = df.merge(data["products"], on="product_id", how="left")
    df = df.merge(data["sellers"], on="seller_id", how="left")
    df = df.merge(data["customers"], on="customer_id", how="left")
    df = df.merge(data["category_translation"], on="product_category_name", how="left")

    # Convert to datetime
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Keep only delivered orders for accurate analysis
    df = df[df["order_status"] == "delivered"].copy()
    df.dropna(
        subset=[
            "order_delivered_customer_date",
            "product_category_name_english",
            "product_weight_g",
        ],
        inplace=True,
    )

    # Feature Engineering
    df["delivery_duration_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["is_late"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)

    # Clean up for modeling
    df = df.drop_duplicates(subset=["order_id", "product_id"])
    df = df.reset_index(drop=True)

    return df
