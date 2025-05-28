import pandas as pd

# 1. Load your existing weekly file
weekly = pd.read_csv("data/weekly_env_data.csv")

# 2. Load the translation + super category mapping
translation = pd.read_csv("data/unique_categories_translated_2.csv", sep=";")

# 3. Merge based on translated name
weekly = weekly.merge(
    translation[['translated_name', 'main_category']],
    how='left',
    left_on='translated_name',
    right_on='translated_name'
)

# 5. Save updated file
weekly.to_csv("data/weekly_env_data.csv", index=False)

print("✅ Updated weekly data with super categories saved: data/weekly_env_data_with_super_category.csv")