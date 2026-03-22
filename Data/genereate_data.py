import numpy as np
import pandas as pd

np.random.seed(42)
N = 10_000


def generate_customers(n):
    card_age = np.random.randint(1, 72, n)

    avg_txn_count = np.random.poisson(lam=15, size=n).astype(float)
    avg_txn_amount = np.random.exponential(scale=60, size=n)
    months_since_last = np.random.choice([0, 1, 2, 3, 4, 5, 6], n,
                                         p=[0.5, 0.2, 0.1, 0.07, 0.06, 0.04, 0.03])
    txn_trend = np.random.uniform(-1, 1, n)
    unique_merchants = np.random.randint(1, 25, n).astype(float)
    international = np.random.binomial(1, 0.2, n)
    complaints = np.random.poisson(lam=0.3, size=n)

    churn_score = (
        (avg_txn_count < 5).astype(float) * 0.3 +
        (months_since_last >= 3).astype(float) * 0.35 +
        (txn_trend < -0.3).astype(float) * 0.2 +
        (unique_merchants < 3).astype(float) * 0.1 +
        (complaints > 1).astype(float) * 0.15 +
        np.random.uniform(0, 0.15, n)
    )

    churned = (churn_score > 0.45).astype(int)

    churned_mask = churned == 1
    avg_txn_count[churned_mask] *= np.random.uniform(0.3, 0.7, churned_mask.sum())
    avg_txn_amount[churned_mask] *= np.random.uniform(0.4, 0.8, churned_mask.sum())

    df = pd.DataFrame({
        "customer_id": range(1001, 1001 + n),
        "card_age_months": card_age,
        "avg_monthly_transactions": np.round(avg_txn_count, 1),
        "avg_transaction_amount": np.round(avg_txn_amount, 2),
        "months_since_last_transaction": months_since_last,
        "transaction_trend": np.round(txn_trend, 3),
        "unique_merchant_count": unique_merchants.astype(int),
        "international_usage": international,
        "complaint_count": complaints,
        "churned": churned
    })

    return df


if __name__ == "__main__":
    df = generate_customers(N)
    df.to_csv("customers.csv", index=False)

    print(f"Dataset yaradıldı: {len(df)} müştəri")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(df.head())