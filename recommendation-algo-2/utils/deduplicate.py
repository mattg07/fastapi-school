import pandas as pd
import re

def standardize_name(name: str) -> str:
    """Convert to lowercase, remove punctuation, strip extra whitespace."""
    # Convert to lowercase
    name = name.lower()
    # Remove punctuation (anything not alphanumeric or whitespace)
    name = re.sub(r"[^\w\s]", "", name)
    # Strip leading/trailing whitespace
    name = name.strip()
    # Collapse any multiple spaces into a single space
    name = re.sub(r"\s+", " ", name)
    return name

input_file = "colleges_data.csv"
output_file = "colleges_data_deduped.csv"

# Read CSV
df = pd.read_csv(input_file)

# Standardize the name column in-place
df["name"] = df["name"].astype(str).apply(standardize_name)

# Remove duplicates based on the now-standardized name
# Keep the first occurrence of each unique name
df.drop_duplicates(subset=["name"], keep="first", inplace=True)

# Save to new CSV
df.to_csv(output_file, index=False)
print(f"De-duplicated CSV saved as {output_file}")
