#this file for check dataset good : bad ratio
import pandas as pd
df = pd.read_csv('data/raw/credit_risk.csv')
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True) * 100)