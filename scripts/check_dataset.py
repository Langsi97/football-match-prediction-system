import pandas as pd

df = pd.read_csv("data/processed/matches_with_rolling_and_form.csv")

print("Shape:", df.shape)
print("\nColumns with rolling:")
print([col for col in df.columns if "roll3" in col][:5])

print("\nForm columns:")
print([col for col in df.columns if "Form" in col])

print("\nHead:")
print(df.head())