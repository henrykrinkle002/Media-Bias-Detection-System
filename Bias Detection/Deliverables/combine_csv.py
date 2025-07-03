import pandas as pd

# Read both CSV files
df1 = pd.read_csv('/Users/amalkurian/Desktop/Dissertation/Bias Detection/nexis_articles2.csv')
df3 = pd.read_csv('/Users/amalkurian/Desktop/Dissertation/Bias Detection/nexis_articles1.csv')

# Concatenate the DataFrames
combined_df = pd.concat([df1, df3], ignore_index=True)

# Remove duplicates (by all columns, or specify subset=['column_name'] for specific columns)
combined_df = combined_df.drop_duplicates(subset = 'title')

# Save to a new CSV file
combined_df.to_csv('/Users/amalkurian/Desktop/Dissertation/Bias Detection/News_Articles/nexis_articles1.csv', index=False)