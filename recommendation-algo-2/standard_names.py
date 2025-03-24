import pandas as pd
import re

def clean_standard_college(value):
    # Remove special characters (keeping alphanumeric and spaces)
    cleaned_value = re.sub(r'[^a-zA-Z0-9 ]', '', value)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
    return cleaned_value

def clean_csv(file_path, output_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Apply cleaning function to 'standard_college' column
    df['standard_college'] = df['standard_college'].astype(str).apply(clean_standard_college)
    
    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to {output_path}")

# Example usage
clean_csv('programs_cleaned.csv', 'programs_cleaned_output.csv')
