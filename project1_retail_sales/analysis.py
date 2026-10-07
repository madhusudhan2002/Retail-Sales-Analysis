import pandas as pd
import numpy as np


# -------------------------------------------------
# 1️⃣ LOAD & BASIC CLEANING
# -------------------------------------------------
def load_data(path):
    """
    Load the Superstore dataset and perform basic cleaning.

    The original Superstore dataset contains columns such as:
    Order ID, Order Date, Customer ID, Product Name, etc.

    These are standardized to the column names used by this project.
    """

    df = pd.read_csv(path)

    # ---------------------------------------------
    # Clean column names
    # ---------------------------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "", regex=False)
    )

    # ---------------------------------------------
    # Rename Superstore columns
    # ---------------------------------------------
    column_mapping = {
        "OrderID": "OrderID",
        "OrderDate": "OrderDate",
        "CustomerID": "CustomerID",
        "CustomerName": "CustomerName",
        "ProductName": "Product",
        "Category": "Category",
        "Sub-Category": "SubCategory",
        "SubCategory": "SubCategory",
        "Segment": "Segment",
        "Region": "Region",
        "State": "State",
        "City": "City",
        "Sales": "Sales",
        "Quantity": "Quantity",
        "Discount": "Discount",
        "Profit": "Profit"
    }

    df.rename(columns=column_mapping, inplace=True)

    # ---------------------------------------------
    # Convert date column
    # ---------------------------------------------
    if "OrderDate" in df.columns:
        df["OrderDate"] = pd.to_datetime(
            df["OrderDate"],
            errors="coerce"
        )

    # Remove rows where date is invalid
    if "OrderDate" in df.columns:
        df.dropna(subset=["OrderDate"], inplace=True)

    # ---------------------------------------------
    # Remove duplicate rows
    # ---------------------------------------------
    df.drop_duplicates(inplace=True)

    # ---------------------------------------------
    # Handle missing numeric values
    # ---------------------------------------------
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_cols:
        df[column] = df[column].fillna(
            df[column].median()
        )

    # ---------------------------------------------
    # Handle missing categorical values
    # ---------------------------------------------
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_cols:
        if df[column].isna().any():
            df[column] = df[column].fillna("Unknown")

    # ---------------------------------------------
    # Ensure Quantity is positive
    # ---------------------------------------------
    if "Quantity" in df.columns:
        df = df[df["Quantity"] > 0]

    # ---------------------------------------------
    # Ensure Sales is valid
    # ---------------------------------------------
    if "Sales" in df.columns:
        df = df[df["Sales"].notna()]

    return df.reset_index(drop=True)


# -------------------------------------------------
# 2️⃣ SALES SUMMARY
# -------------------------------------------------
def sales_summary(df):
    """
    Generate basic sales summaries.
    """

    total_revenue = df["Sales"].sum()

    monthly_sales = (
        df.resample(
            "ME",
            on="OrderDate"
        )["Sales"]
        .sum()
    )

    category_sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    return total_revenue, monthly_sales, category_sales


# -------------------------------------------------
# 3️⃣ RFM ANALYSIS + SCORING
# -------------------------------------------------
def rfm_analysis(df):
    """
    Perform Recency, Frequency and Monetary analysis.

    Recency  = Days since customer's last purchase
    Frequency = Number of unique orders
    Monetary = Total customer sales
    """

    max_date = df["OrderDate"].max()

    # ---------------------------------------------
    # Create RFM table
    # ---------------------------------------------
    rfm = df.groupby("CustomerID").agg(
        Recency=(
            "OrderDate",
            lambda x: (max_date - x.max()).days
        ),
        Frequency=(
            "OrderID",
            "nunique"
        ),
        Monetary=(
            "Sales",
            "sum"
        )
    )

    # ---------------------------------------------
    # Safe RFM scoring
    # ---------------------------------------------

    # Recency:
    # Lower recency = better customer
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"),
        5,
        labels=[5, 4, 3, 2, 1]
    )

    # Frequency:
    # Higher frequency = better customer
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    )

    # Monetary:
    # Higher spending = better customer
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    )

    # ---------------------------------------------
    # Convert scores to integers
    # ---------------------------------------------
    rfm["R_Score"] = rfm["R_Score"].astype(int)
    rfm["F_Score"] = rfm["F_Score"].astype(int)
    rfm["M_Score"] = rfm["M_Score"].astype(int)

    # ---------------------------------------------
    # Overall RFM score
    # ---------------------------------------------
    rfm["RFM_Score"] = (
        rfm["R_Score"]
        + rfm["F_Score"]
        + rfm["M_Score"]
    )

    return rfm


# -------------------------------------------------
# 4️⃣ ADVANCED FEATURE ENGINEERING
# -------------------------------------------------
def advanced_feature_engineering(df):
    """
    Create time-based and derived business features.
    """

    df = df.copy()

    # Sort by date
    df = df.sort_values("OrderDate").reset_index(drop=True)

    # ---------------------------------------------
    # Time-based features
    # ---------------------------------------------
    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["MonthName"] = df["OrderDate"].dt.month_name()

    df["Day"] = df["OrderDate"].dt.day
    df["Weekday"] = df["OrderDate"].dt.weekday

    df["WeekdayName"] = (
        df["OrderDate"].dt.day_name()
    )

    df["Quarter"] = (
        df["OrderDate"].dt.quarter
    )

    # Weekend flag
    df["IsWeekend"] = (
        df["Weekday"] >= 5
    ).astype(int)

    # ---------------------------------------------
    # Rolling 7-order average sales by category
    # ---------------------------------------------
    df["Rolling_7Day_Sales"] = (
        df.groupby("Category")["Sales"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

    # ---------------------------------------------
    # Average price per unit
    # ---------------------------------------------
    df["AvgPrice"] = np.where(
        df["Quantity"] > 0,
        df["Sales"] / df["Quantity"],
        0
    )

    # ---------------------------------------------
    # Profit margin
    # ---------------------------------------------
    if "Profit" in df.columns:
        df["ProfitMargin"] = np.where(
            df["Sales"] != 0,
            (df["Profit"] / df["Sales"]) * 100,
            0
        )

    return df


# -------------------------------------------------
# 5️⃣ REGION × CATEGORY PIVOT TABLE
# -------------------------------------------------
def region_category_pivot(df):
    """
    Create a region vs category sales pivot table.
    """

    pivot = pd.pivot_table(
        df,
        values="Sales",
        index="Region",
        columns="Category",
        aggfunc="sum",
        fill_value=0
    )

    return pivot


# -------------------------------------------------
# 6️⃣ CORRELATION MATRIX
# -------------------------------------------------
def correlation_matrix(df):
    """
    Generate correlation matrix for numerical variables.
    """

    return df.corr(numeric_only=True)


# -------------------------------------------------
# 7️⃣ EDA SUMMARY STATISTICS
# -------------------------------------------------
def eda_summary(df):
    """
    Generate an overall exploratory data analysis summary.
    """

    summary = {
        "Shape": df.shape,

        "Missing Values": (
            df.isnull()
            .sum()
            .to_dict()
        ),

        "Data Types": (
            df.dtypes
            .astype(str)
            .to_dict()
        ),

        "Describe": (
            df.describe(
                include="all"
            )
            .to_dict()
        ),

        "Unique Customers": (
            df["CustomerID"].nunique()
        ),

        "Unique Products": (
            df["Product"].nunique()
        ),

        "Unique Orders": (
            df["OrderID"].nunique()
        ),

        "Unique Categories": (
            df["Category"].nunique()
        )
    }

    return summary


# -------------------------------------------------
# 8️⃣ OUTLIER REMOVAL — IQR METHOD
# -------------------------------------------------
def remove_outliers_iqr(df, column):
    """
    Remove outliers from a numerical column
    using the Interquartile Range method.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    filtered_df = df[
        (df[column] >= lower)
        & (df[column] <= upper)
    ].copy()

    return filtered_df


# -------------------------------------------------
# 9️⃣ MONTHLY GROWTH RATE
# -------------------------------------------------
def calculate_monthly_growth(df):
    """
    Calculate month-over-month sales growth.
    """

    monthly_sales = (
        df.resample(
            "ME",
            on="OrderDate"
        )["Sales"]
        .sum()
    )

    growth = (
        monthly_sales
        .pct_change()
        * 100
    )

    return growth


# -------------------------------------------------
# 🔟 PARETO ANALYSIS — 80/20 RULE
# -------------------------------------------------
def pareto_analysis(df):
    """
    Analyze how much of total revenue is generated
    by the highest-value customers.
    """

    customer_sales = (
        df.groupby("CustomerID")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    total_sales = customer_sales.sum()

    if total_sales == 0:
        return {
            "top_customers_count": 0,
            "total_customers": len(customer_sales),
            "percentage_of_customers": 0
        }

    cumulative_percentage = (
        customer_sales.cumsum()
        / total_sales
        * 100
    )

    # Number of customers required to generate
    # approximately 80% of revenue
    top_customers_count = (
        cumulative_percentage
        .le(80)
        .sum()
    )

    total_customers = len(customer_sales)

    percentage_of_customers = (
        top_customers_count
        / total_customers
        * 100
        if total_customers > 0
        else 0
    )

    return {
        "top_customers_count": int(
            top_customers_count
        ),
        "total_customers": int(
            total_customers
        ),
        "percentage_of_customers": round(
            percentage_of_customers,
            2
        )
    }
