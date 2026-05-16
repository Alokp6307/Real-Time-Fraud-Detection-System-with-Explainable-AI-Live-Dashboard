import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Fraud Operations Dashboard",
    page_icon="🚨",
    layout="wide",
)


@st.cache_data(show_spinner="Loading CSV files...")
def load_data():
    transaction = pd.read_csv("train_transaction.csv")
    identity = pd.read_csv("train_identity.csv")
    data = transaction.merge(identity, on="TransactionID", how="left")
    data["HourOfDay"] = (data["TransactionDT"] // 3600) % 24
    return data


@st.cache_data(show_spinner="Preparing model data...")
def prepare_model_data(data):
    df = data.copy()

    df["AmtToMeanRatio"] = df["TransactionAmt"] / df["TransactionAmt"].mean()

    if "DeviceType" in df.columns and "DeviceInfo" in df.columns:
        df["DeviceRisk"] = np.where(
            df["DeviceType"].isna()
            | df["DeviceInfo"].isna()
            | (df["DeviceType"].astype(str).str.lower() == "mobile"),
            1,
            0,
        )
    else:
        df["DeviceRisk"] = 0

    missing_percent = df.isna().mean() * 100
    keep_columns = missing_percent[missing_percent <= 50].index.tolist()
    df = df[keep_columns]

    y = df["isFraud"]
    transaction_ids = df["TransactionID"]
    X = df.drop(columns=["isFraud", "TransactionID"])

    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in numeric_columns:
        X[column] = X[column].fillna(X[column].median())

    for column in categorical_columns:
        mode_value = X[column].mode()
        fill_value = mode_value.iloc[0] if len(mode_value) else "Unknown"
        X[column] = X[column].fillna(fill_value).astype(str).astype("category").cat.codes

    X = X.loc[:, ~X.columns.duplicated()]
    return X, y, transaction_ids


@st.cache_resource(show_spinner="Training LightGBM model...")
def train_model(X, y):
    sample_size = min(120000, len(X))
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        random_state=42,
        stratify=y,
    )

    model = LGBMClassifier(
        random_state=42,
        n_estimators=250,
        learning_rate=0.05,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_sample, y_sample)
    return model


@st.cache_data(show_spinner="Scoring transactions...")
def score_transactions(_model, X, data):
    probabilities = _model.predict_proba(X)[:, 1]

    scored = data.copy()
    scored["Fraud_Probability"] = probabilities
    scored["Risk_Tier"] = np.select(
        [
            scored["Fraud_Probability"] >= 0.75,
            scored["Fraud_Probability"] >= 0.40,
        ],
        ["Critical Risk", "Suspicious"],
        default="Clear",
    )
    return scored


def apply_sidebar_filters(scored):
    st.sidebar.header("Filters")

    selected_tiers = st.sidebar.multiselect(
        "Risk tier",
        ["Critical Risk", "Suspicious", "Clear"],
        default=["Critical Risk", "Suspicious", "Clear"],
    )

    min_amt = float(scored["TransactionAmt"].min())
    max_amt = float(scored["TransactionAmt"].max())
    amount_range = st.sidebar.slider(
        "Transaction amount",
        min_value=min_amt,
        max_value=max_amt,
        value=(min_amt, max_amt),
    )

    fraud_filter = st.sidebar.selectbox(
        "Actual fraud label",
        ["All", "Fraud only", "Non-fraud only"],
    )

    filtered = scored[
        scored["Risk_Tier"].isin(selected_tiers)
        & scored["TransactionAmt"].between(amount_range[0], amount_range[1])
    ]

    if fraud_filter == "Fraud only":
        filtered = filtered[filtered["isFraud"] == 1]
    elif fraud_filter == "Non-fraud only":
        filtered = filtered[filtered["isFraud"] == 0]

    return filtered


def show_overview(filtered):
    st.title("Fraud Operations Dashboard")

    total_transactions = len(filtered)
    total_fraud = int(filtered["isFraud"].sum())
    detection_rate = filtered["Fraud_Probability"].mean() * 100
    avg_fraud_amount = filtered.loc[filtered["isFraud"] == 1, "TransactionAmt"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{total_transactions:,}")
    col2.metric("Total fraud count", f"{total_fraud:,}")
    col3.metric("Detection rate", f"{detection_rate:.2f}%")
    col4.metric("Average fraud amount", f"{avg_fraud_amount:,.2f}")

    left, right = st.columns(2)

    tier_counts = filtered["Risk_Tier"].value_counts().reset_index()
    tier_counts.columns = ["Risk_Tier", "Count"]
    fig_tier = px.bar(
        tier_counts,
        x="Risk_Tier",
        y="Count",
        color="Risk_Tier",
        title="Transactions by Risk Tier",
    )
    left.plotly_chart(fig_tier, use_container_width=True)

    hourly = (
        filtered.groupby(["HourOfDay", "Risk_Tier"])
        .size()
        .reset_index(name="Transaction_Count")
    )
    fig_hour = px.line(
        hourly,
        x="HourOfDay",
        y="Transaction_Count",
        color="Risk_Tier",
        markers=True,
        title="Hour-of-Day Pattern",
    )
    right.plotly_chart(fig_hour, use_container_width=True)

    amount_summary = (
        filtered.groupby("Risk_Tier")["TransactionAmt"].mean().reset_index()
    )
    fig_amount = px.bar(
        amount_summary,
        x="Risk_Tier",
        y="TransactionAmt",
        color="Risk_Tier",
        title="Average Transaction Amount by Risk Tier",
    )
    st.plotly_chart(fig_amount, use_container_width=True)


def show_transaction_explorer(filtered):
    st.title("Transaction Explorer")

    transaction_id = st.text_input("Search TransactionID")
    table_data = filtered.copy()

    if transaction_id:
        table_data = table_data[
            table_data["TransactionID"].astype(str).str.contains(transaction_id)
        ]

    display_columns = [
        "TransactionID",
        "isFraud",
        "Fraud_Probability",
        "Risk_Tier",
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "DeviceType",
        "HourOfDay",
    ]
    display_columns = [col for col in display_columns if col in table_data.columns]

    st.dataframe(
        table_data[display_columns].sort_values(
            "Fraud_Probability", ascending=False
        ),
        use_container_width=True,
        height=520,
    )

    st.subheader("Live Risk Score")
    selected_id = st.number_input(
        "Enter TransactionID",
        min_value=int(filtered["TransactionID"].min()),
        max_value=int(filtered["TransactionID"].max()),
        step=1,
    )

    selected_row = filtered[filtered["TransactionID"] == selected_id]
    if not selected_row.empty:
        row = selected_row.iloc[0]
        st.metric("Fraud probability", f"{row['Fraud_Probability']:.2%}")
        st.write("Risk tier:", row["Risk_Tier"])
    else:
        st.warning("TransactionID not found in current filters.")


def explain_in_plain_english(shap_row, feature_values):
    top_indices = np.argsort(np.abs(shap_row.values))[-5:][::-1]
    lines = []

    for index in top_indices:
        feature = shap_row.feature_names[index]
        value = feature_values[feature]
        direction = "increased" if shap_row.values[index] > 0 else "decreased"
        lines.append(
            f"{feature} = {value} {direction} the fraud probability."
        )

    return lines


def show_shap_explainer(scored, X, model):
    st.title("SHAP Explainer")

    transaction_id = st.number_input(
        "Enter TransactionID for SHAP explanation",
        min_value=int(scored["TransactionID"].min()),
        max_value=int(scored["TransactionID"].max()),
        step=1,
        key="shap_transaction_id",
    )

    matched_index = scored.index[scored["TransactionID"] == transaction_id].tolist()

    if not matched_index:
        st.warning("TransactionID not found.")
        return

    row_index = matched_index[0]
    X_one = X.loc[[row_index]]
    probability = model.predict_proba(X_one)[0, 1]

    st.metric("Fraud probability", f"{probability:.2%}")
    st.write("Risk tier:", scored.loc[row_index, "Risk_Tier"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_one)

    st.subheader("SHAP Waterfall Plot")
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Plain-English Explanation")
    for line in explain_in_plain_english(shap_values[0], X_one.iloc[0]):
        st.write("-", line)


data = load_data()
X, y, transaction_ids = prepare_model_data(data)
model = train_model(X, y)
scored_data = score_transactions(model, X, data)
filtered_data = apply_sidebar_filters(scored_data)

page = st.sidebar.radio(
    "Pages",
    ["Overview", "Transaction Explorer", "SHAP Explainer"],
)

if page == "Overview":
    show_overview(filtered_data)
elif page == "Transaction Explorer":
    show_transaction_explorer(filtered_data)
else:
    show_shap_explainer(scored_data, X, model)
