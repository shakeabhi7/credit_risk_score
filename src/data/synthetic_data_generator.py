import os
import random
import numpy as np
import pandas as pd
from faker import Faker

from src.monitoring.logger import setup_logger

# Logger
logger = setup_logger("synthetic_data_generator")

# Config
DEFAULT_OUTPUT_PATH = "data/raw/credit_risk.csv"
FAKER_LOCALE = "en_IN"
RANDOM_SEED = 42


def generate_synthetic_data(
    n_samples: int = 5000,
    output_path: str = DEFAULT_OUTPUT_PATH,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:

    logger.info("Starting raw synthetic data generation")
    logger.info(f"Samples: {n_samples}")
    logger.info(f"Output path: {output_path}")

    # Reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    Faker.seed(random_seed)

    fake = Faker(FAKER_LOCALE)
    records = []

    for i in range(n_samples):
        age = random.randint(18, 80)
        income = random.randint(20_000, 150_000)

        employment_years = random.randint(0, max(age - 18, 0))

        if employment_years == 0:
            employment_type = "Unemployed"
        else:
            employment_type = random.choices(
                ["Salaried", "Self-Employed"],
                weights=[0.7, 0.3]
            )[0]

        debt = random.randint(0, int(income * 2))
        credit_limit = random.randint(5_000, 50_000)
        credit_used = random.randint(0, credit_limit)

        # these varaible only for target calculation ---
        debt_to_income = debt / (income + 1)
        credit_utilization = credit_used / (credit_limit + 1)

        risk_score = np.clip(
            debt_to_income * 0.5 +
            credit_utilization * 0.3 +
            (1 - employment_years / 40) * 0.2,
            0,
            1
        )

        # BETTER - More realistic
        target = 1 if risk_score > 0.98 + random.uniform(-0.05, 0.06) else 0
        # Result: ~20% Bad, ~80% Good

        records.append({
            "customer_id": f"CUST_{i:05d}",
            "age": age,
            "income": income,
            "debt": debt,
            "credit_limit": credit_limit,
            "credit_used": credit_used,
            "employment_years": employment_years,
            "employment_type": employment_type,
            "target": target,
        })

        if (i + 1) % 1000 == 0:
            logger.info(f"Generated {i + 1}/{n_samples} records")

    df = pd.DataFrame(records)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    good = (df["target"] == 0).sum()
    bad = (df["target"] == 1).sum()

    logger.info("Raw data generation completed.")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Good credit (0): {good}")
    logger.info(f"Bad credit (1): {bad}")

    if bad > 0:
        logger.info(f"Imbalance ratio (good:bad) = {good / bad:.2f}:1")
    else:
        logger.warning("No bad credit samples generated")

    logger.info(f"Saved file: {output_path}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    generate_synthetic_data(n_samples=5000)
