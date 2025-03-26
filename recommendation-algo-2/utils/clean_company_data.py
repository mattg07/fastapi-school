import pandas as pd
import re
import unicodedata

# Load CSV file into a DataFrame
input_file = 'companies_data.csv'  # Replace with your actual file path
df = pd.read_csv(input_file)

# Function to standardize school names
def standardize_name(name):
    name = name.lower()  # lowercase
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('ascii')  # normalize special characters
    name = re.sub(r'[^\w\s]', '', name)  # remove punctuation and special characters, preserve spaces
    name = re.sub(r'\s+', ' ', name)  # replace multiple spaces with single space
    return name.strip()

# Standardize all school columns
school_columns = [col for col in df.columns if col.startswith('school_')]
for col in school_columns:
    df[col] = df[col].astype(str).apply(standardize_name)

# Save cleaned DataFrame to a new CSV file
output_file = 'companies_data_cleaned.csv'  # Specify your desired output file name
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")
