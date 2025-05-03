import pandas as pd

# Load datasets
train_df = pd.read_csv("us_births.csv")
test_df = pd.read_csv("us_births_test.csv")

print(train_df.shape)
train_df.head()

# Check for null values and data types
print(train_df.isnull().sum())
print(train_df.info())

# Column conversion

# Change true/false to 1/0
train_df['mother_risk_factor'] = train_df['mother_risk_factor'].astype(bool).astype(int)

# Change categorical values to 1/0
train_df['mother_diabetes_gestational'] = train_df['mother_diabetes_gestational'].map({'Y': 1, 'N': 0})

# Change sex to 1/0
train_df['newborn_sex'] = train_df['newborn_sex'].map({'F': 0, 'M': 1})

X_train = train_df.drop("newborn_birth_weight", axis=1)
y_train = train_df["newborn_birth_weight"]

X_test = test_df.drop("newborn_birth_weight", axis=1)
y_test = test_df["newborn_birth_weight"]
