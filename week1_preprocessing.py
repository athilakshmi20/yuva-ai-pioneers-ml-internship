
import pandas as pd
import matplotlib.pyplot as plt

# Load the original student dataset
df = pd.read_csv("student-mat.csv", sep=";")

print("FIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nSUMMARY STATISTICS")
print(df.describe())


# STEP 2: DATA PREPROCESSING

# Create a working copy of the original dataset
cleaned_df = df.copy()

# Check for missing values
print("\nMissing values before cleaning:",
      cleaned_df.isnull().sum().sum())

# Handle missing values only if they exist
for col in cleaned_df.columns:
    if cleaned_df[col].isnull().any():
        if cleaned_df[col].dtype == "object":
            cleaned_df[col] = cleaned_df[col].fillna(
                cleaned_df[col].mode()[0]
            )
        else:
            cleaned_df[col] = cleaned_df[col].fillna(
                cleaned_df[col].median()
            )

# Remove duplicate rows if any exist
cleaned_df = cleaned_df.drop_duplicates()

print("Missing values after cleaning:",
      cleaned_df.isnull().sum().sum())
print("Duplicate rows after cleaning:",
      cleaned_df.duplicated().sum())

# STEP 3: CATEGORICAL ENCODING

categorical_cols = cleaned_df.select_dtypes(
    include=["object"]
).columns.tolist()

encoded_df = pd.get_dummies(
    cleaned_df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

print("\nDataset shape after encoding:", encoded_df.shape)
print("Categorical columns encoded:", len(categorical_cols))

# STEP 4: NORMALIZATION

from sklearn.preprocessing import MinMaxScaler

# Scale numerical input features, excluding final grade columns
grade_cols = ["G1", "G2", "G3"]

numeric_cols = encoded_df.select_dtypes(
    include=["number"]
).columns.tolist()

features_to_scale = [
    col for col in numeric_cols if col not in grade_cols
]

scaler = MinMaxScaler()
encoded_df[features_to_scale] = scaler.fit_transform(
    encoded_df[features_to_scale]
)

print("\nNormalization completed.")
print(encoded_df[features_to_scale].describe().round(2))

# Save the preprocessed dataset
encoded_df.to_csv("student_performance_preprocessed.csv",
                  index=False)

print("\nPreprocessed dataset saved successfully!")




# Distribution of final grades
plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=11, edgecolor="black")
plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("final_grade_distribution.png")
plt.show()

# Average final grade by study time
average_grades = df.groupby("studytime")["G3"].mean()

plt.figure(figsize=(8, 5))
average_grades.plot(kind="bar", edgecolor="black")
plt.title("Average Final Grade by Study Time")
plt.xlabel("Weekly Study Time Category")
plt.ylabel("Average Final Grade")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("average_grade_by_studytime.png")
plt.show()

