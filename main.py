from project1_retail_sales.analysis import (
    load_data,
    advanced_feature_engineering,
    region_category_pivot,
    rfm_analysis
)

from project1_retail_sales.statistics_analysis import (
    t_test_weekend_sales
)

from project1_retail_sales.modeling import (
    churn_prediction,
    random_forest_churn,
    sales_regression_model,
    shap_analysis
)

from project1_retail_sales.visualization import (
    plot_monthly_sales,
    plot_category_sales,
    plot_correlation_heatmap,
    plot_sales_distribution,
    plot_sales_boxplot,
    plot_pivot_heatmap,
    plot_roc_curve,
    plot_feature_importance
)

from project1_retail_sales.executive_summary import (
    generate_executive_summary
)


def main():

    # =================================================
    # 1. LOAD DATA
    # =================================================

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = load_data(
        "project1_retail_sales/data/retail_sales.csv"
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print(
        f"Unique Customers: "
        f"{df['CustomerID'].nunique():,}"
    )

    print(
        f"Unique Orders: "
        f"{df['OrderID'].nunique():,}"
    )

    print(
        f"Unique Products: "
        f"{df['Product'].nunique():,}"
    )

    print(
        f"Date Range: "
        f"{df['OrderDate'].min().date()} "
        f"to "
        f"{df['OrderDate'].max().date()}"
    )


    # =================================================
    # 2. FEATURE ENGINEERING
    # =================================================

    print("\n" + "=" * 60)
    print("PERFORMING FEATURE ENGINEERING")
    print("=" * 60)

    df = advanced_feature_engineering(df)

    print(
        f"Feature-engineered dataset shape: "
        f"{df.shape}"
    )


    # =================================================
    # 3. REGION × CATEGORY PIVOT
    # =================================================

    print("\n" + "=" * 60)
    print("REGION × CATEGORY ANALYSIS")
    print("=" * 60)

    pivot = region_category_pivot(df)

    print("\nRegion-Category Sales:")
    print(pivot)


    # =================================================
    # 4. STATISTICAL ANALYSIS
    # =================================================

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    stats_results = t_test_weekend_sales(df)

    if "error" in stats_results:
        print(
            "Statistical analysis error:",
            stats_results["error"]
        )

    else:
        for key, value in stats_results.items():
            print(f"{key}: {value}")


    # =================================================
    # 5. SALES SUMMARIES
    # =================================================

    print("\n" + "=" * 60)
    print("SALES SUMMARY")
    print("=" * 60)

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

    print("\nTotal Revenue:")
    print(f"${df['Sales'].sum():,.2f}")

    print("\nCategory Sales:")
    print(category_sales)

    print("\nMonthly Sales:")
    print(monthly_sales)


    # =================================================
    # 6. VISUALIZATIONS
    # =================================================

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_monthly_sales(monthly_sales)

    plot_category_sales(category_sales)

    plot_correlation_heatmap(df)

    plot_sales_distribution(df)

    plot_sales_boxplot(df)

    plot_pivot_heatmap(pivot)


    # =================================================
    # 7. RFM ANALYSIS
    # =================================================

    print("\n" + "=" * 60)
    print("RFM ANALYSIS")
    print("=" * 60)

    rfm = rfm_analysis(df)

    print(
        f"Customers analyzed: {len(rfm):,}"
    )

    print("\nTop RFM Customers:")

    print(
        rfm.sort_values(
            "RFM_Score",
            ascending=False
        ).head(10)
    )


    # =================================================
    # 8. MACHINE LEARNING
    # =================================================

    print("\n" + "=" * 60)
    print("MACHINE LEARNING")
    print("=" * 60)

    print(
        "\nNOTE:"
        "\nThe current churn model will be replaced"
        "\nwith a proper future-based churn definition."
        "\nML training is temporarily disabled."
    )

    logistic_results = {
        "error": "Churn model temporarily disabled - target leakage will be fixed in the next step."
    }

    rf_results = {
        "error": "Churn model temporarily disabled - target leakage will be fixed in the next step."
    }

    # -------------------------------------------------
    # Sales regression
    # -------------------------------------------------

    print("\nTraining Sales Regression Model...")

    reg_results = sales_regression_model(df)


    # =================================================
    # 9. EXECUTIVE SUMMARY
    # =================================================

    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    generate_executive_summary(
        df,
        regression_results=reg_results,
        logistic_results=None,
        rf_results=None
    )


    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
