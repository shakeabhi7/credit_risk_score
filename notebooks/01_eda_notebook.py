import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.monitoring.logger import setup_logger

logger = setup_logger('eda')
logger.info("Loading data")

loader = DataLoader()
df = loader.load_csv('data/raw/credit_risk.csv')

logger.info(f"Data shape:{df.shape}")
print("first few rows:")
print(df.head(5))

# Missing Values
logger.info("-"*5)
logger.info("Missing Values handling")
logger.info("-"*5)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count':missing,
    'Percentage':missing_percent
})
logger.info(f"{missing_df[missing_df['Missing_Count'] > 0]}")


#Target Distribution
logger.info("-"*5)
logger.info("Target Distribution")
logger.info("-"*5)

target_counts = df['target'].value_counts()
target_percent = df['target'].value_counts(normalize=True)*100


logger.info(f"Good Credit (0): {target_counts[0]} ({target_percent[0]:.2f}%)")
logger.info(f"Bad Credit (1): {target_counts[1]} ({target_percent[1]:.2f}%)")

# #visualization
plt.figure(figsize=(6,4))

sns.countplot(
    x='target',
    data=df,
    hue='target',          
    palette='pastel',     
    legend=False           
)

plt.title('Target Distribution - Good vs Bad Credit')
plt.xlabel('Credit Status (0=Good,1=Bad)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('notebooks/visuals/target_distribution.png',
            dpi=100,
            bbox_inches='tight')
logger.info("Saved: 01_target_distribution.png")
plt.close()

# Numerical Features Analysis

logger.info("-"*6)
logger.info("NUMERICAL FEATURES ANALYSIS")
logger.info("-"*6)

numeric_features = ['age', 'income', 'debt', 'credit_limit', 'credit_used', 'employment_years']

for col in numeric_features:
    logger.info(f"{col}:")
    logger.info(f"Min: {df[col].min()}, Max: {df[col].max()}")
    logger.info(f"  Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
    logger.info(f"  Std: {df[col].std():.2f}")
    logger.info(f"  Skewness: {df[col].skew():.2f}")


# Distibution plots
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

palette = sns.color_palette("plasma", len(numeric_features))

for idx, col in enumerate(numeric_features):
    sns.histplot(
        df[col],
        bins=20,
        kde=True,              
        ax=axes[idx],
        color=palette[idx]
    )
    
    axes[idx].set_title(f'Distribution of {col}',fontsize=10)
    axes[idx].set_xlabel(col,fontsize=9)
    axes[idx].set_ylabel('Frequency',fontsize=9)

plt.tight_layout(pad=2)
plt.savefig('notebooks/visuals/numerical_distributions.png', dpi=100, bbox_inches='tight')
logger.info("Saved: numerical_distributions.png")
# plt.show()
plt.close()



# CATEGORICAL FEATURES ANALYSIS

logger.info("-"*5)
logger.info("CATEGORICAL FEATURES ANALYSIS")
logger.info("-"*5)

categorical_features = ['employment_type']
for col in categorical_features:
    logger.info(f"\n{col}:")
    logger.info(f"{df[col].value_counts()}")


#Visualization
plt.figure(figsize=(10, 6))

sns.countplot(
    x='employment_type',
    data=df,
    hue='employment_type',
    palette='Set2',
    legend=False
)

plt.title('Employment Type Distribution', fontsize=14)
plt.xlabel('Employment Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig('notebooks/visuals/employment_type.png', dpi=100, bbox_inches='tight')
logger.info("Saved: employment_type.png")
plt.show()
plt.close()

# CORRELATION ANALYSIS

logger.info("-"*5)
logger.info("CORRELATION ANALYSIS")
logger.info("-"*5)

correlation = df[numeric_features + ['target']].corr()
logger.info(f"Correlation with target:")
print(correlation['target'].sort_values(ascending=True))

#heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('notebooks/visuals/correlation_matrix.png', dpi=100, bbox_inches='tight')
logger.info("Saved: correlation_matrix.png")
plt.show()
plt.close()


# AGE ANALYSIS

logger.info("-"*5)
logger.info("AGE ANALYSIS")
logger.info("-"*5)

age_by_target = df.groupby('target')['age'].agg(['mean', 'median', 'min', 'max'])
logger.info(f"\nAge by Credit Status:\n{age_by_target}")


plt.figure(figsize=(10,5))

sns.histplot(
    data=df,
    x='age',
    hue='target',             
    bins=30,
    kde=True,                  
    palette='Set1',
    alpha=0.5,
    element='step'             
)

plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution by Credit Status')
plt.tight_layout()

plt.savefig('notebooks/visuals/age_by_target.png', dpi=100, bbox_inches='tight')
logger.info("Saved: age_by_target.png")
plt.show()
plt.close()


# INCOME ANALYSIS

logger.info("-"*5)
logger.info("INCOME ANALYSIS")
logger.info("-"*5)

income_by_target = df.groupby('target')['income'].agg(['mean', 'median', 'min', 'max'])
logger.info(f"\nIncome by Credit Status:\n{income_by_target}")


plt.figure(figsize=(10,5))

sns.histplot(
    data=df,
    x='income',
    hue='target',          
    bins=30,
    kde=True,               
    palette='deep',
    alpha=0.5,
    element='step'         
)

plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income Distribution by Credit Status')
plt.tight_layout()
plt.savefig('notebooks/visuals/income_by_target.png', dpi=100, bbox_inches='tight')
logger.info("Saved: income_by_target.png")
plt.show()
plt.close()



# DEBT ANALYSIS

logger.info("-"*5)
logger.info("DEBT ANALYSIS")
logger.info("-"*5)

debt_by_target = df.groupby('target')['debt'].agg(['mean', 'median', 'min', 'max'])
logger.info(f"\nDebt by Credit Status:\n{debt_by_target}")


plt.figure(figsize=(10,5))

sns.histplot(
    data=df,
    x='debt',
    hue='target',        
    bins=30,
    kde=True,            
    palette='viridis',
    alpha=0.5,
    element='step'       
)

plt.xlabel('Debt')
plt.ylabel('Frequency')
plt.title('Debt Distribution by Credit Status')
plt.tight_layout()
plt.savefig('notebooks/visuals/debt_by_target.png', dpi=100, bbox_inches='tight')
logger.info("Saved: debt_by_target.png")
plt.show()
plt.close()


# DEBT-TO-INCOME RATIO ANALYSIS

logger.info("-"*5)
logger.info("DEBT-TO-INCOME RATIO ANALYSIS")
logger.info("-"*5)

df['debt_to_income'] = df['debt'] / (df['income'] + 1)

dti_by_target = df.groupby('target')['debt_to_income'].agg(['mean', 'median', 'min', 'max'])
logger.info(f"\nDebt-to-Income Ratio by Credit Status:\n{dti_by_target}")

plt.figure(figsize=(10,5))


sns.histplot(
    data=df,
    x='debt_to_income',
    hue='target',
    bins=30,
    kde=True,
    palette='tab10',     
    alpha=0.5,
    element='step'
)

plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Frequency')
plt.title('Debt-to-Income Ratio by Credit Status')
plt.tight_layout()

plt.savefig('notebooks/visuals/debt_to_income.png', dpi=100, bbox_inches='tight')
logger.info("Saved: debt_to_income.png")
plt.show()
plt.close()


# CREDIT UTILIZATION ANALYSIS

logger.info("-"*5)
logger.info("CREDIT UTILIZATION ANALYSIS")
logger.info( "-"*5)

df['credit_utilization'] = df['credit_used'] / (df['credit_limit'] + 1)

cu_by_target = df.groupby('target')['credit_utilization'].agg(['mean', 'median', 'min', 'max'])
logger.info(f"\nCredit Utilization by Credit Status:\n{cu_by_target}")


plt.figure(figsize=(10,5))

sns.histplot(
    data=df,
    x='credit_utilization',
    hue='target',
    bins=30,
    kde=True,
    palette='magma',   
    alpha=0.5,
    element='step'
)

plt.xlabel('Credit Utilization')
plt.ylabel('Frequency')
plt.title('Credit Utilization by Credit Status')
plt.tight_layout()
plt.savefig('notebooks/visuals/credit_utilization.png', dpi=100, bbox_inches='tight')
logger.info("Saved: credit_utilization.png")
plt.show()
plt.close()


# EMPLOYMENT YEARS ANALYSIS

logger.info("-"*5)
logger.info("EMPLOYMENT YEARS ANALYSIS")
logger.info("-"*5)

emp_by_target = df.groupby('target')['employment_years'].agg(['mean', 'median', 'min', 'max'])
logger.info(f"\nEmployment Years by Credit Status:\n{emp_by_target}")


plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x='employment_years',
    hue='target',
    bins=30,
    kde=True,
    palette='colorblind',
    alpha=0.5,
    element='step'
)

plt.xlabel('Employment Years')
plt.ylabel('Frequency')
plt.title('Employment Years by Credit Status')

plt.tight_layout()
plt.savefig('notebooks/visuals/employment_years.png', dpi=100, bbox_inches='tight')
logger.info("Saved: employment_years.png")
plt.show()
plt.close()


# KEY INSIGHTS

logger.info("-"*5)
logger.info("KEY INSIGHTS FROM EDA")
logger.info("-"*5)

logger.info("Data Quality:")
logger.info(f"   - Total records: {len(df)}")
logger.info(f"   - Missing values: {df.isnull().sum().sum()}")
logger.info(f"   - Duplicates: {df.duplicated().sum()}")

logger.info("Target Distribution:")
logger.info(f"   - Good Credit (0): {target_percent[0]:.1f}%")
logger.info(f"   - Bad Credit (1): {target_percent[1]:.1f}%")
logger.info(f"   - Class Imbalance: {target_counts[0]/target_counts[1]:.2f}:1")

logger.info("Key Correlations with Target:")
for col, corr in correlation['target'].nlargest(5).items():
    if col != 'target':
        logger.info(f"   - {col}: {corr:.3f}")

logger.info("\nBad Credit Characteristics:")
logger.info(f"   - Higher debt-to-income: {dti_by_target.loc[1, 'mean']:.3f} vs {dti_by_target.loc[0, 'mean']:.3f}")
logger.info(f"   - Higher credit util: {cu_by_target.loc[1, 'mean']:.3f} vs {cu_by_target.loc[0, 'mean']:.3f}")
logger.info(f"   - Lower employment stability: {emp_by_target.loc[1, 'mean']:.1f} vs {emp_by_target.loc[0, 'mean']:.1f} years")

logger.info("-"*5)
logger.info(" EDA COMPLETE!")
logger.info("All visualizations saved in notebooks/visuals/ folder")
logger.info("-" *5)