
import sqlite3
import random
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. APP CONFIGURATION AND DATABASE
# --------------------------------------------------

st.set_page_config(
    page_title="Expense Analyzer AI",
    page_icon="💰",
    layout="wide"
)

DB_PATH = "expenses.db"

CATEGORIES = [
    "Food", "Transport", "Shopping", "Rent",
    "Bills", "Entertainment", "Health",
    "Education", "Travel", "Other"
]

conn = sqlite3.connect(DB_PATH, check_same_thread=False)

conn.execute("""
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    expense_date TEXT NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL CHECK(amount > 0),
    description TEXT
)
""")
conn.commit()


def load_expenses():
    df = pd.read_sql_query(
        "SELECT * FROM expenses ORDER BY expense_date",
        conn
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["expense_date"])
        df["amount"] = pd.to_numeric(df["amount"])
    return df


# --------------------------------------------------
# 2. SIDEBAR
# --------------------------------------------------

st.sidebar.title("💰 Expense Analyzer")
st.sidebar.caption("Personal finance powered by ML")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Add Expense",
        "Expense History",
        "ML Predictions"
    ]
)

st.sidebar.info(
    "Algorithms: Linear Regression, "
    "Random Forest, K-Means"
)

# --------------------------------------------------
# 3. ADD EXPENSE
# --------------------------------------------------

if page == "Add Expense":
    st.title("📝 Add Daily Expense")
    st.write("Record your spending to build your personal history.")

    with st.form("expense_form"):
        expense_date = st.date_input(
            "Expense date", value=date.today()
        )
        category = st.selectbox("Category", CATEGORIES)
        amount = st.number_input(
            "Amount (₹)",
            min_value=1.0,
            max_value=10_000_000.0,
            value=100.0,
            step=10.0
        )
        description = st.text_input(
            "Description",
            placeholder="Example: Lunch with friends"
        )

        submitted = st.form_submit_button(
            "Save Expense", type="primary"
        )

        if submitted:
            conn.execute(
                """
                INSERT INTO expenses
                (expense_date, category, amount, description)
                VALUES (?, ?, ?, ?)
                """,
                (
                    expense_date.isoformat(),
                    category,
                    amount,
                    description.strip()
                )
            )
            conn.commit()
            st.success("Expense saved successfully!")
            st.rerun()

# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

df = load_expenses()

if page == "Expense History":
    st.title("📒 Expense History")

    if df.empty:
        st.info("No expenses yet. Add your first expense.")
    else:
        st.dataframe(
            df[["id", "date", "category", "amount", "description"]],
            use_container_width=True,
            hide_index=True
        )

        csv_data = df[
            ["date", "category", "amount", "description"]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Expenses as CSV",
            data=csv_data,
            file_name="my_expenses.csv",
            mime="text/csv"
        )

        selected_id = st.selectbox(
            "Select an expense ID to delete",
            df["id"].tolist()
        )

        if st.button("Delete Selected Expense"):
            conn.execute(
                "DELETE FROM expenses WHERE id = ?",
                (int(selected_id),)
            )
            conn.commit()
            st.success("Expense deleted.")
            st.rerun()

# --------------------------------------------------
# 5. PREPARE MONTHLY DATA
# --------------------------------------------------

def monthly_totals(data):
    result = data.copy()
    result["month"] = result["date"].dt.to_period("M")

    return (
        result.groupby("month")["amount"]
        .sum()
        .sort_index()
        .asfreq("M", fill_value=0)
        .rename("total")
        .reset_index()
    )


# --------------------------------------------------
# 6. DASHBOARD AND VISUALIZATIONS
# --------------------------------------------------

if page == "Dashboard":
    st.title("📊 Personalized Expense Dashboard")
    st.caption("Understand where your money goes.")

    if df.empty:
        st.info("Add some expenses to see your dashboard.")
    else:
        current_month = pd.Timestamp.today().to_period("M")
        df["month"] = df["date"].dt.to_period("M")

        current_spend = df.loc[
            df["month"] == current_month, "amount"
        ].sum()

        total_spend = df["amount"].sum()
        avg_daily = df.groupby("date")["amount"].sum().mean()

        previous_month = current_month - 1
        previous_spend = df.loc[
            df["month"] == previous_month, "amount"
        ].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("All-Time Spending", f"₹{total_spend:,.0f}")
        c2.metric("This Month", f"₹{current_spend:,.0f}")

        change = None
        if previous_spend > 0:
            change = (
                (current_spend - previous_spend)
                / previous_spend * 100
            )

        c3.metric(
            "Vs Previous Month",
            f"{change:+.1f}%" if change is not None else "N/A",
            delta=f"{change:+.1f}%" if change is not None else None,
            delta_color="inverse"
        )
        c4.metric("Avg. Spend per Active Day", f"₹{avg_daily:,.0f}")

        left, right = st.columns(2)

        with left:
            st.subheader("Spending by Category")
            category_data = (
                df.groupby("category", as_index=False)["amount"]
                .sum()
                .sort_values("amount", ascending=False)
            )
            fig = px.pie(
                category_data,
                names="category",
                values="amount",
                hole=0.45,
                title="All-Time Category Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Daily Spending")
            daily = (
                df.groupby("date", as_index=False)["amount"]
                .sum()
            )
            fig = px.line(
                daily,
                x="date",
                y="amount",
                markers=True,
                title="Expense Trend by Day"
            )
            fig.update_yaxes(title="Amount (₹)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Spending Comparison")
        mdata = monthly_totals(df)
        mdata["month"] = mdata["month"].astype(str)

        fig = px.bar(
            mdata,
            x="month",
            y="total",
            text_auto=".2s",
            title="Total Expenses per Month"
        )
        fig.update_yaxes(title="Amount (₹)")
        st.plotly_chart(fig, use_container_width=True)

        if previous_spend > 0:
            st.subheader("Personalized Insight")
            difference = current_spend - previous_spend
            pct = difference / previous_spend * 100

            if difference > 0:
                st.warning(
                    f"You have spent ₹{difference:,.0f} more "
                    f"than last month ({pct:.1f}% increase)."
                )
            elif difference < 0:
                st.success(
                    f"You have spent ₹{-difference:,.0f} less "
                    f"than last month ({abs(pct):.1f}% decrease)."
                )
            else:
                st.info("Your spending matches last month.")

# --------------------------------------------------
# 7. ML PREDICTION: LINEAR REGRESSION + RANDOM FOREST
# --------------------------------------------------

def build_features(monthly):
    """
    Predict each month's expenditure using only
    information available before that month.
    """
    values = monthly["total"].astype(float).tolist()
    periods = monthly["month"].tolist()

    X, y = [], []

    for i in range(1, len(values)):
        previous = values[i - 1]
        avg3 = sum(values[max(0, i - 3):i]) / len(
            values[max(0, i - 3):i]
        )

        target_month = periods[i]
        X.append([
            i,
            previous,
            avg3,
            target_month.month
        ])
        y.append(values[i])

    return X, y


if page == "ML Predictions":
    st.title("🤖 Next-Month Expense Prediction")
    st.write(
        "The system learns from your previous monthly expenses "
        "and compares two regression algorithms."
    )

    if df.empty:
        st.info("Add expenses before generating predictions.")
    else:
        monthly = monthly_totals(df)
        monthly["month"] = pd.PeriodIndex(monthly["month"], freq="M")

        st.subheader("Historical Monthly Expenses")

        history_chart = monthly.copy()
        history_chart["month"] = history_chart["month"].astype(str)

        st.plotly_chart(
            px.line(
                history_chart,
                x="month",
                y="total",
                markers=True,
                title="Monthly Spending History"
            ).update_yaxes(title="Amount (₹)"),
            use_container_width=True
        )

        if len(monthly) < 6:
            st.warning(
                f"You have {len(monthly)} months of recorded data. "
                "Record at least 6 different months for this "
                "basic prediction model. More history is better."
            )
        else:
            X, y = build_features(monthly)

            # Train both models on historical observations.
            linear_model = LinearRegression()
            forest_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=1,
                random_state=42
            )

            linear_model.fit(X, y)
            forest_model.fit(X, y)

            values = monthly["total"].astype(float).tolist()
            last_month = monthly["month"].iloc[-1]
            next_month = last_month + 1
            i = len(values)

            last3 = values[-3:]
            next_features = [[
                i,
                values[-1],
                sum(last3) / len(last3),
                next_month.month
            ]]

            lr_prediction = max(
                0.0, float(linear_model.predict(next_features)[0])
            )
            rf_prediction = max(
                0.0, float(forest_model.predict(next_features)[0])
            )

            a, b = st.columns(2)
            a.metric(
                f"Linear Regression — {next_month}",
                f"₹{lr_prediction:,.0f}"
            )
            b.metric(
                f"Random Forest — {next_month}",
                f"₹{rf_prediction:,.0f}"
            )

            st.info(
                "These are estimates, not guaranteed future expenses. "
                "They use your recorded monthly totals and historical "
                "trends; sudden life changes are not known to the model."
            )

            comparison = pd.DataFrame({
                "Model": ["Linear Regression", "Random Forest"],
                "Predicted Expense (₹)": [
                    lr_prediction, rf_prediction
                ]
            })

            st.plotly_chart(
                px.bar(
                    comparison,
                    x="Model",
                    y="Predicted Expense (₹)",
                    text_auto=".2f",
                    title="Prediction Model Comparison"
                ),
                use_container_width=True
            )

            st.subheader("Historical vs Predicted")
            forecast = pd.DataFrame({
                "Month": [
                    str(p) for p in monthly["month"]
                ] + [str(next_month)],
                "Expenses (₹)": values + [None],
                "Linear Regression Forecast (₹)": (
                    [None] * len(values) + [lr_prediction]
                ),
                "Random Forest Forecast (₹)": (
                    [None] * len(values) + [rf_prediction]
                )
            })

            st.dataframe(forecast, use_container_width=True)

# --------------------------------------------------
# 8. KAGGLE DATASET: ML TRAINING AND EVALUATION
# --------------------------------------------------

if page == "ML Predictions":
    st.divider()
    st.header("Kaggle Dataset: Model Training")

    DATASET_PATH = "data/expenses_dataset.csv"

    try:
        dataset = pd.read_csv(DATASET_PATH)

        required_columns = [
            "DATE & Time",
            "Category",
            "Amount",
            "Payment Mode"
        ]

        missing = [
            col for col in required_columns
            if col not in dataset.columns
        ]

        if missing:
            st.error(f"Missing CSV columns: {missing}")
        else:
            dataset["date"] = pd.to_datetime(
                dataset["DATE & Time"],
                errors="coerce"
            )
            dataset["amount"] = pd.to_numeric(
                dataset["Amount"],
                errors="coerce"
            )

            dataset = dataset.dropna(
                subset=[
                    "date", "amount",
                    "Category", "Payment Mode"
                ]
            )

            dataset = dataset[dataset["amount"] > 0].copy()
            dataset = dataset.sort_values("date")

            st.write(
                f"Valid transactions loaded: **{len(dataset)}**"
            )
            st.write(
                f"Date range: {dataset['date'].min().date()} "
                f"to {dataset['date'].max().date()}"
            )

            if len(dataset) < 20:
                st.warning(
                    "Not enough valid transactions for a "
                    "meaningful train/test demonstration."
                )
            else:
                # Predict transaction amount using information
                # available for each transaction.
                dataset["month_number"] = dataset["date"].dt.month
                dataset["day_of_week"] = (
                    dataset["date"].dt.dayofweek
                )
                dataset["day_of_month"] = dataset["date"].dt.day

                feature_columns = [
                    "Category",
                    "Payment Mode",
                    "month_number",
                    "day_of_week",
                    "day_of_month"
                ]

                X = dataset[feature_columns]
                y = dataset["amount"]

                # Chronological split: train on earlier records,
                # evaluate on the latest 20%.
                split_at = int(len(dataset) * 0.8)

                X_train = X.iloc[:split_at]
                X_test = X.iloc[split_at:]
                y_train = y.iloc[:split_at]
                y_test = y.iloc[split_at:]

                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.pipeline import Pipeline
                from sklearn.metrics import (
                    mean_absolute_error,
                    mean_squared_error,
                    r2_score
                )

                categorical_features = [
                    "Category", "Payment Mode"
                ]
                numeric_features = [
                    "month_number",
                    "day_of_week",
                    "day_of_month"
                ]

                preprocessor = ColumnTransformer([
                    (
                        "categories",
                        OneHotEncoder(handle_unknown="ignore"),
                        categorical_features
                    ),
                    ("numbers", "passthrough", numeric_features)
                ])

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        min_samples_leaf=3
                    )
                }

                results = []
                trained_models = {}

                for model_name, estimator in models.items():
                    pipeline = Pipeline([
                        ("preprocessing", preprocessor),
                        ("model", estimator)
                    ])

                    pipeline.fit(X_train, y_train)
                    predictions = pipeline.predict(X_test)

                    mae = mean_absolute_error(
                        y_test, predictions
                    )
                    rmse = mean_squared_error(
                        y_test, predictions
                    ) ** 0.5

                    results.append({
                        "Model": model_name,
                        "MAE (₹)": round(mae, 2),
                        "RMSE (₹)": round(rmse, 2),
                        "R²": round(
                            r2_score(y_test, predictions), 3
                        )
                    })

                    trained_models[model_name] = pipeline

                st.subheader("Held-out Test Results")
                st.caption(
                    "Models are evaluated on the latest 20% of "
                    "transactions, not the records used for training. "
                    "These metrics measure transaction-amount "
                    "prediction, not next-month spending."
                )
                st.dataframe(
                    pd.DataFrame(results),
                    hide_index=True,
                    use_container_width=True
                )

                st.subheader("Try a Transaction Prediction")

                col1, col2 = st.columns(2)

                with col1:
                    selected_category = st.selectbox(
                        "Category",
                        sorted(dataset["Category"].unique())
                    )
                    selected_payment = st.selectbox(
                        "Payment Mode",
                        sorted(dataset["Payment Mode"].unique())
                    )

                with col2:
                    selected_date = st.date_input(
                        "Transaction date",
                        value=date.today()
                    )
                    selected_model = st.selectbox(
                        "Model",
                        list(trained_models.keys())
                    )

                if st.button("Predict Transaction Amount"):
                    input_data = pd.DataFrame([{
                        "Category": selected_category,
                        "Payment Mode": selected_payment,
                        "month_number": selected_date.month,
                        "day_of_week": selected_date.weekday(),
                        "day_of_month": selected_date.day
                    }])

                    predicted_amount = trained_models[
                        selected_model
                    ].predict(input_data)[0]

                    st.metric(
                        "Estimated transaction amount",
                        f"₹{max(0, predicted_amount):,.2f}"
                    )

                st.info(
                    "This is a demonstration model. The dataset "
                    "has limited date coverage, and the prediction "
                    "does not use description or personal income. "
                    "Results should not be treated as guaranteed "
                    "spending estimates."
                )

    except FileNotFoundError:
        st.error(
            "Dataset not found. Ensure the CSV exists at "
            "data/expenses_dataset.csv."
        )
    except Exception as error:
        st.error(f"Dataset training failed: {error}")

# --------------------------------------------------
# 8. K-MEANS: SPENDING BEHAVIOR CLUSTERS
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Spending Behavior")

if not df.empty:
    category_spend = (
        df.groupby("category", as_index=False)["amount"]
        .sum()
    )

    if len(category_spend) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(
            category_spend[["amount"]]
        )

        kmeans = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )
        category_spend["cluster"] = kmeans.fit_predict(scaled)

        centers = scaler.inverse_transform(
            kmeans.cluster_centers_
        ).flatten()

        # Give clusters understandable names based on their centers.
        ranked_clusters = sorted(
            range(3), key=lambda c: centers[c]
        )
        names = {
            ranked_clusters[0]: "Low spending",
            ranked_clusters[1]: "Medium spending",
            ranked_clusters[2]: "High spending"
        }
        category_spend["Spending Pattern"] = (
            category_spend["cluster"].map(names)
        )

        st.sidebar.write(
            "K-Means groups categories by their recorded "
            "total spending."
        )
        st.sidebar.dataframe(
            category_spend[
                ["category", "amount", "Spending Pattern"]
            ].rename(columns={
                "category": "Category",
                "amount": "Total (₹)"
            }),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.sidebar.caption(
            "Record expenses in at least 3 categories to run K-Means."
        )
else:
    st.sidebar.caption("Add expenses to analyze spending patterns.")
