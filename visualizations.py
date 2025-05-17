import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("TRAIN.csv")

# Drop NA and non-numeric columns for correlation and pairplot
df_clean = df.dropna()
df_numeric = df_clean.select_dtypes(include=['int64', 'float64'])

# Pairplot
sns.pairplot(df_numeric)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# Heatmap
plt.figure(figsize=(12, 8))
corr = df_numeric.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
