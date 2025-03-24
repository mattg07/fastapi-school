import pandas as pd
import re

def clean_rankings(value):
    # Remove phrases like "Best Colleges for X in America", keeping only the ranking number
    return re.sub(r'\d+ Best Colleges in America', lambda m: m.group(0).split()[0], str(value)) if isinstance(value, str) else value

def clean_csv(file_path, output_path):
    # Load the CSV file with low_memory=False to avoid mixed types warning
    df = pd.read_csv(file_path, low_memory=False, keep_default_na=False)
    
    # Apply cleaning function to all ranking-related columns
    df = df.applymap(clean_rankings)
    
    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to {output_path}")

# Example usage
clean_csv('universities_out.csv', 'universities_out2.csv')
