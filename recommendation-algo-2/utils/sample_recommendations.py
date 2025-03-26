import pandas as pd
import numpy as np
import random
import sys
import os
from tabulate import tabulate
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from generate_recommendations2 import recommend_schools
except ImportError as e:
    print(f"Error importing recommend_schools: {e}")
    sys.exit(1)

def format_recommendations(recommendations, program, gpa, sat):
    """Format recommendations into a readable table"""
    if recommendations is None or recommendations.empty:
        return f"\nNo recommendations found for {program} (GPA: {gpa}, SAT: {sat})\n"
    
    # Select relevant columns and rename for display
    display_cols = [
        'School', 'Recommendation_Tier', 'Median_Earnings_1yr',
        'Avg_GPA', 'Avg_SAT', 'Admission_Rate'
    ]
    
    display_df = recommendations[display_cols].copy()
    
    # Format numeric columns
    display_df['Median_Earnings_1yr'] = display_df['Median_Earnings_1yr'].apply(
        lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
    )
    display_df['Avg_GPA'] = display_df['Avg_GPA'].apply(
        lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
    )
    display_df['Avg_SAT'] = display_df['Avg_SAT'].apply(
        lambda x: f"{x:.0f}" if pd.notnull(x) else "N/A"
    )
    display_df['Admission_Rate'] = display_df['Admission_Rate'].apply(
        lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A"
    )
    
    # Rename columns for display
    display_df.columns = ['School', 'Tier', 'Earnings (1yr)', 'Avg GPA', 'Avg SAT', 'Admit Rate']
    
    # Create header
    header = f"\nRecommendations for {program}"
    subheader = f"Student Profile - GPA: {gpa:.2f}, SAT: {sat}"
    separator = "=" * max(len(header), len(subheader))
    
    # Format table
    table = tabulate(
        display_df,
        headers='keys',
        tablefmt='grid',
        showindex=False,
        numalign='right',
        stralign='left'
    )
    
    return f"\n{header}\n{separator}\n{subheader}\n\n{table}\n"

def sample_combinations(n_samples=10):
    """Generate random GPA/SAT combinations"""
    gpas = np.round(np.random.uniform(2.5, 4.0, n_samples), 2)
    sats = np.random.randint(1000, 1601, n_samples)
    return list(zip(gpas, sats))

def main():
    # Load programs
    try:
        programs = sorted(pd.read_csv("programs_cleaned.csv")["program"].unique())
    except Exception as e:
        print(f"Error reading programs file: {e}")
        sys.exit(1)
    
    # Sample programs and combinations
    n_samples = 10
    sampled_programs = random.sample(list(programs), min(n_samples, len(programs)))
    combinations = sample_combinations(n_samples)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"sample_recommendations_{timestamp}.txt"
    
    print(f"\nGenerating {n_samples} sample recommendations...")
    print("This might take a few minutes...")
    
    with open(output_file, 'w') as f:
        for program, (gpa, sat) in zip(sampled_programs, combinations):
            try:
                recommendations = recommend_schools(
                    user_gpa=gpa,
                    user_sat=sat,
                    user_program=program,
                    verbose=False
                )
                
                # Format and save results
                result_text = format_recommendations(recommendations, program, gpa, sat)
                f.write(result_text)
                f.write("\n" + "-"*80 + "\n")
                
                # Also print to console
                print(result_text)
                print("-"*80)
                
            except Exception as e:
                error_msg = f"Error processing {program} (GPA: {gpa}, SAT: {sat}): {str(e)}"
                print(error_msg)
                f.write(error_msg + "\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 