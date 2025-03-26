import csv
import re

# Paths to your input and output CSV files
input_csv_path = "school_sup_data.csv"   
output_csv_path = "school_sup_data_cleaned.csv"  

def clean_institution_name(name: str) -> str:
    """
    Takes an institution name string and:
      1. Converts to lowercase.
      2. Removes any non-alphanumeric and non-space characters (dashes, ampersands, punctuation, etc.).
      3. Collapses multiple spaces into one.
      4. Removes leading or trailing spaces.
    """
    # 1. lowercase
    name = name.lower()
    # 2. remove any non-alphanumeric and non-space characters
    name = re.sub(r'[^a-z0-9\s]', '', name)
    # 3. collapse multiple spaces into one
    name = re.sub(r'\s+', ' ', name)
    # 4. strip leading/trailing spaces
    name = name.strip()
    return name

# Open the input CSV, read it, and write out to the output CSV
with open(input_csv_path, mode='r', encoding='utf-8', newline='') as infile, \
     open(output_csv_path, mode='w', encoding='utf-8', newline='') as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    # Create a writer object based on the same fieldnames as the input
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # Clean the "INSTNM" column using our function
        if row.get('INSTNM'):
            row['INSTNM'] = clean_institution_name(row['INSTNM'])
        
        # Write the modified row to the output
        writer.writerow(row)

print(f"Done! Cleaned CSV written to: {output_csv_path}")
