import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer
from src.monitoring.logger import setup_logger

logger = setup_logger('feature_analysis')

logger.info("Loading data and creating features...")
loader = DataLoader()
df = loader.load_csv('data/processed/credit_risk_featured.csv')

engineer = FeatureEngineer()
df = engineer.create_features(df)

logger.info(f"Total features: {len(df.columns)}")
logger.info(f"Engineered features: {len(engineer.get_feature_names())}")


# ENGINEERED FEATURES ANALYSIS

logger.info("-"*60)
logger.info("ENGINEERED FEATURES ANALYSIS")
logger.info("-"*60)
engineered_features = engineer.get_feature_names()
for feature in engineered_features:
    if feature != 'age_group':  # Skip categorical
        logger.info(f"\n{feature}:")
        logger.info(f"  Min: {df[feature].min():.4f}, Max: {df[feature].max():.4f}")
        logger.info(f"  Mean: {df[feature].mean():.4f}, Median: {df[feature].median():.4f}")
        logger.info(f"  Std: {df[feature].std():.4f}")


# DEBT-TO-INCOME ANALYSIS

logger.info("-"*6)
logger.info("DEBT-TO-INCOME RATIO ANALYSIS")
logger.info("-"*6)

dti_by_target = df.groupby('target')['debt_to_income'].agg(['mean','median','min','max'])
logger.info(f"Debt-to-Income by Credit Status:\n{dti_by_target}")

mean_good = dti_by_target.loc[0, 'mean']
mean_bad = dti_by_target.loc[1, 'mean']
plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x='debt_to_income',
    hue='target',
    bins=30,
    kde=True,
    palette={0: "green", 1: "red"},
    alpha=0.5,
    element='step'
)

# Mean lines
plt.axvline(mean_good, color='green', linestyle='--', linewidth=2,
            label=f'Mean Good: {mean_good:.2f}')

plt.axvline(mean_bad, color='red', linestyle='--', linewidth=2,
            label=f'Mean Bad: {mean_bad:.2f}')

plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Frequency')
plt.title('Debt-to-Income Distribution by Credit Status')

plt.legend(title='Credit Status')

plt.tight_layout()

plt.savefig('notebooks/feature_visuals/debt_to_income_engineered.png', dpi=200, bbox_inches='tight')
logger.info("Saved: debt_to_income_engineered.png")
plt.show()
plt.close()


# CREDIT UTILIZATION ANALYSIS

logger.info("-"*6)
logger.info("CREDIT UTILIZATION ANALYSIS")
logger.info("-"*6)


plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x='credit_utilization',
    hue='target',
    bins=30,
    kde=True,
    palette={0: "green", 1: "red"},
    alpha=0.5,
    element="step"
)

plt.axvline(mean_good, color='green', linestyle='--', linewidth=2,
            label=f'Mean Good: {mean_good:.2f}')
plt.axvline(mean_bad, color='red', linestyle='--', linewidth=2,
            label=f'Mean Bad: {mean_bad:.2f}')

plt.title("Credit Utilization Distribution by Credit Status")
plt.xlabel("Credit Utilization")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
cu_by_target = df.groupby('target')['credit_utilization'].agg(['mean', 'median', 'min', 'max'])
logger.info(f"Credit Utilization by Credit Status:\n{cu_by_target}")

plt.savefig('notebooks/feature_visuals/credit_utilization_engineered.png', dpi=100, bbox_inches='tight')
logger.info("Saved: credit_utilization_engineered.png")
plt.show()
plt.close()



# EMPLOYMENT STABILITY ANALYSIS

logger.info("-"*6)
logger.info("EMPLOYMENT STABILITY ANALYSIS")
logger.info("-"*6)

es_by_target = df.groupby('target')['employment_stability'].agg(['mean', 'median', 'min', 'max'])
logger.info(f"Employment Stability by Credit Status:\n{es_by_target}")


mean_good = es_by_target.loc[0, 'mean']
mean_bad = es_by_target.loc[1, 'mean']

plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x='employment_stability',
    hue='target',
    bins=30,
    kde=True,
    palette={0: "green", 1: "red"},
    alpha=0.5,
    element="step"
)

plt.axvline(mean_good, color='green', linestyle='--', linewidth=2,
            label=f'Mean Good: {mean_good:.2f}')
plt.axvline(mean_bad, color='red', linestyle='--', linewidth=2,
            label=f'Mean Bad: {mean_bad:.2f}')

plt.title("Employment Stability Distribution by Credit Status")
plt.xlabel("Employment Stability")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig('notebooks/feature_visuals/employment_stability.png', dpi=100, bbox_inches='tight')
logger.info("Saved: employment_stability.png")
plt.show()
plt.close()

# ENGINEERED FEATURES CORRELATION WITH TARGET

logger.info("-"*6)
logger.info("ENGINEERED FEATURES CORRELATION WITH TARGET")
logger.info("-"*6)

# Select only numeric engineered features
numeric_engineered = [f for f in engineered_features if f != 'age_group']
correlation = df[numeric_engineered + ['target']].corr()


target_corr = correlation['target'].drop('target').sort_values()

logger.info("\nCorrelation with target:")
for feature, corr_value in target_corr.items():
    logger.info(f"  {feature}: {corr_value:.4f}")

plt.figure(figsize=(10, 8))

sns.barplot(
    x=target_corr.values,
    y=target_corr.index,
    hue=target_corr.index,
    palette="viridis"
)

plt.title("Engineered Features Correlation with Target")
plt.xlabel("Correlation")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("notebooks/feature_visuals/engineered_features_correlation.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
logger.info("Saved: engineered_features_correlation.png")


# AGE GROUP DISTRIBUTION

logger.info("-"*6)
logger.info("AGE GROUP ANALYSIS")
logger.info("-"*6)

age_group_dist = df['age_group'].value_counts().sort_index()
logger.info(f"Age Group Distribution:\n{age_group_dist}")

age_group_target = pd.crosstab(df['age_group'], df['target'])
logger.info(f"Age Group vs Target:\n{age_group_target}")

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x='age_group',
    hue='target',
    palette={0: "green", 1: "red"}
)

plt.title("Credit Status by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("notebooks/feature_visuals/age_group_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

logger.info("Saved: age_group_distribution.png")


# FEATURE RELATIONSHIPS

logger.info("-"*6)
logger.info("FEATURE RELATIONSHIPS")
logger.info("-"*6)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='income', y='debt_to_income',
                hue='target', palette={0: "green", 1: "red"}, alpha=0.6)
plt.title("Debt-to-Income vs Income")
plt.tight_layout()
plt.savefig("notebooks/feature_visuals/dti_vs_income.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='credit_limit', y='credit_utilization',
                hue='target', palette={0: "green", 1: "red"}, alpha=0.6)
plt.title("Credit Utilization vs Credit Limit")
plt.tight_layout()
plt.savefig("notebooks/feature_visuals/util_vs_limit.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='age', y='employment_stability',
                hue='target', palette={0: "green", 1: "red"}, alpha=0.6)
plt.title("Employment Stability vs Age")
plt.tight_layout()
plt.savefig("notebooks/feature_visuals/employment_vs_age.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='debt', y='debt_impact',
                hue='target', palette={0: "green", 1: "red"}, alpha=0.6)
plt.title("Debt Impact vs Debt")
plt.tight_layout()
plt.savefig("notebooks/feature_visuals/debtimpact_vs_debt.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

logger.info(" Saved: Feature relationship plots")


# SUMMARY

logger.info("-"*6)
logger.info(" FEATURE ANALYSIS COMPLETE!")
logger.info("-"*6)
logger.info("Created 6 visualization files:")
logger.info("debt_to_income_engineered.png")
logger.info("credit_utilization_engineered.png")
logger.info("employment_stability.png")
logger.info("engineered_features_correlation.png")
logger.info("age_group_distribution.png")
logger.info("feature_relationships.png")
logger.info("-"*6)