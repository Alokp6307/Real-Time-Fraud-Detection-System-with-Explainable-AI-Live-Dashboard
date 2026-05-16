# Fraud Operations Dashboard

Streamlit dashboard for fraud risk segmentation, transaction exploration, and SHAP-based model explanation.

## Run Locally

```bash
streamlit run app.py
```

## Pages

- Overview: total transactions, fraud count, detection rate, average fraud amount, and interactive charts.
- Transaction Explorer: searchable and filterable transaction table with live risk score by TransactionID.
- SHAP Explainer: enter a TransactionID to view a SHAP waterfall plot and plain-English explanation.

## Streamlit Community Cloud Deployment

1. Upload this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Select the repository.
4. Set the main file path as `app.py`.
5. Deploy the app.

## Live URL

Add Streamlit Community Cloud live URL here after deployment:
