import pandas as pd
import numpy as np

def numeric_or_nan(x):
    try:
        return float(x)
    except:
        return np.nan

def recommend_schools(
    user_gpa: float,
    user_sat: int,
    user_program: str,
    path_colleges: str = "colleges_data_cleaned.csv",
    path_programs: str = "programs_cleaned.csv",
    path_school_sup: str = "school_sup_data.csv",
    path_companies: str = "companies_data_cleaned.csv",
    verbose: bool = False
):
    if verbose:
        print(f"\nLoading data for program: {user_program}")
    
    df_colleges = pd.read_csv(path_colleges)
    df_programs = pd.read_csv(path_programs)
    df_school_sup = pd.read_csv(path_school_sup)
    df_companies = pd.read_csv(path_companies)

    if verbose:
        print(f"Loaded data files:")
        print(f"- Colleges: {len(df_colleges)} rows")
        print(f"- Programs: {len(df_programs)} rows")
        print(f"- School Sup: {len(df_school_sup)} rows")
        print(f"- Companies: {len(df_companies)} rows")

    df_colleges["average_sat_composite"] = pd.to_numeric(df_colleges["average_sat_composite"], errors="coerce")
    df_school_sup["SAT_AVG"] = pd.to_numeric(df_school_sup["SAT_AVG"], errors="coerce")

    df_colleges["name_clean"] = df_colleges["name"].str.lower().str.strip()
    if "standard_college" in df_programs.columns:
        df_programs["school_name_clean"] = df_programs["standard_college"].str.lower().str.strip()
    else:
        df_programs["school_name_clean"] = df_programs["name"].str.lower().str.strip()
    df_school_sup["school_name_clean"] = df_school_sup["INSTNM"].str.lower().str.strip()

    school_cols = [col for col in df_companies.columns if col.startswith("school_")]
    school_to_companies = {}
    for idx, row in df_companies.iterrows():
        company_name = row["company"]
        for sc in school_cols:
            school_raw = row[sc]
            if pd.notnull(school_raw):
                school_clean = str(school_raw).lower().strip()
                if school_clean not in school_to_companies:
                    school_to_companies[school_clean] = set()
                school_to_companies[school_clean].add(company_name)

    user_program_clean = user_program.lower().strip()
    if "program_clean" in df_programs.columns:
        df_prog_filtered = df_programs[df_programs["program_clean"].str.lower() == user_program_clean]
    else:
        df_prog_filtered = df_programs[df_programs["program"].str.lower() == user_program_clean]

    if verbose:
        print(f"\nProgram filtering results:")
        print(f"Programs matching '{user_program}': {len(df_prog_filtered)}")

    if df_prog_filtered.empty:
        if verbose:
            print(f"No schools found offering program '{user_program}'.")
        return pd.DataFrame()

    merged = pd.merge(
        df_prog_filtered,
        df_colleges,
        left_on="school_name_clean",
        right_on="name_clean",
        how="inner",
        suffixes=("_prog", "_col")
    )

    if verbose:
        print(f"After merging with colleges: {len(merged)} rows")

    if "name_col" in merged.columns:
        merged.rename(columns={"name_col": "name"}, inplace=True)

    merged = pd.merge(
        merged,
        df_school_sup,
        on="school_name_clean",
        how="inner"
    )

    if verbose:
        print(f"After merging with school sup: {len(merged)} rows")

    def get_sat_value(row):
        if not pd.isnull(row["average_sat_composite"]):
            return row["average_sat_composite"]
        return row["SAT_AVG"]

    merged["sat_combined"] = merged.apply(get_sat_value, axis=1)

    def gpa_tier(row):
        if pd.isnull(row["average_gpa"]):
            return None
        diff = row["average_gpa"] - user_gpa
        if abs(diff) <= 0.3:
            return "Match"
        elif 0.3 < diff <= 0.6:
            return "Safety"
        elif -0.6 <= diff < -0.3:
            return "Reach"
        else:
            return None

    def sat_tier(row):
        if pd.isnull(row["sat_combined"]):
            return None
        diff = row["sat_combined"] - user_sat
        if abs(diff) <= 150:
            return "Match"
        elif 150 < diff <= 300:
            return "Safety"
        elif -300 <= diff < -150:
            return "Reach"
        else:
            return None

    merged["gpa_tier"] = merged.apply(gpa_tier, axis=1)
    merged["sat_tier"] = merged.apply(sat_tier, axis=1)

    def get_combined_tier(row):
        gpa = row["gpa_tier"]
        sat = row["sat_tier"]
        if gpa == "Match" and sat == "Match":
            return "Match"
        elif "Safety" in (gpa, sat) and "Reach" not in (gpa, sat):
            return "Safety"
        elif "Reach" in (gpa, sat) and "Safety" not in (gpa, sat):
            return "Reach"
        else:
            return None

    merged["tier"] = merged.apply(get_combined_tier, axis=1)

    final_candidates = merged[merged["tier"].notnull()].copy()

    if final_candidates.empty:
        if verbose:
            print("No schools are within the specified GPA/SAT range for that program.")
            print("Applying fallback strategy...")

        fallback_pool = merged.copy()
        fallback_pool["data_score"] = (
            fallback_pool["earn_mdn_1yr"].notna().astype(int) +
            fallback_pool["earn_mdn_5yr"].notna().astype(int) +
            fallback_pool["average_gpa"].notna().astype(int) +
            fallback_pool["average_sat_composite"].notna().astype(int) +
            fallback_pool["acceptance_rate"].notna().astype(int)
        )
        fallback_pool = fallback_pool.sort_values(
            by=["data_score", "earn_mdn_1yr"],
            ascending=[False, False]
        )
        fallback_top = fallback_pool.head(5).copy()
        fallback_top["tier"] = "Suggested (Fallback)"
        final_candidates = fallback_top

    def parse_accept_rate(x):
        if isinstance(x, str) and x.endswith("%"):
            try:
                return float(x.replace("%", "")) / 100.0
            except:
                return np.nan
        try:
            return float(x)
        except:
            return np.nan

    def parse_dollar_amount(x):
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "")
        try:
            return float(x)
        except:
            return np.nan

    final_candidates["acceptance_rate_numeric"] = final_candidates["acceptance_rate"].apply(parse_accept_rate)
    final_candidates["average_net_price_numeric"] = final_candidates["average_net_price"].apply(parse_dollar_amount)
    final_candidates["earn_mdn_1yr_num"] = final_candidates["earn_mdn_1yr"].apply(numeric_or_nan)
    final_candidates["earn_mdn_5yr_num"] = final_candidates["earn_mdn_5yr"].apply(numeric_or_nan)

    def get_companies_for_school(row):
        school_name_clean = row["name_clean"]
        return list(school_to_companies.get(school_name_clean, []))

    final_candidates["fortune_500_companies"] = final_candidates.apply(get_companies_for_school, axis=1)

    if "name" not in final_candidates.columns:
        if verbose:
            print("Error: Could not find 'name' column after merges.")
        return pd.DataFrame()

    def calculate_academic_distance(row, user_gpa, user_sat):
        gpa_distance = 0
        sat_distance = 0
        count = 0
        
        if pd.notnull(row["average_gpa"]) and pd.notnull(user_gpa):
            gpa_distance = abs(row["average_gpa"] - user_gpa) / 4.0  # Normalize to 0-1 scale
            count += 1
            
        if pd.notnull(row["sat_combined"]) and pd.notnull(user_sat):
            sat_distance = abs(row["sat_combined"] - user_sat) / 1600  # Normalize to 0-1 scale
            count += 1
            
        if count == 0:
            return None
        
        return (gpa_distance + sat_distance) / count if count > 0 else None

    def calculate_earnings_score(row):
        earnings_1yr = row["earn_mdn_1yr_num"] if pd.notnull(row["earn_mdn_1yr_num"]) else 0
        earnings_5yr = row["earn_mdn_5yr_num"] if pd.notnull(row["earn_mdn_5yr_num"]) else 0
        
        # Normalize earnings to 0-1 scale based on the dataset
        max_1yr = final_candidates["earn_mdn_1yr_num"].max() or 1
        max_5yr = final_candidates["earn_mdn_5yr_num"].max() or 1
        
        score_1yr = earnings_1yr / max_1yr if max_1yr > 0 else 0
        score_5yr = earnings_5yr / max_5yr if max_5yr > 0 else 0
        
        # Weight 5-year earnings slightly more
        if earnings_1yr > 0 and earnings_5yr > 0:
            return (0.4 * score_1yr + 0.6 * score_5yr)
        elif earnings_1yr > 0:
            return score_1yr * 0.8  # Penalty for missing 5-year data
        elif earnings_5yr > 0:
            return score_5yr * 0.8  # Penalty for missing 1-year data
        return None

    def calculate_data_completeness(row):
        # Key metrics we care about
        metrics = [
            "average_gpa", "sat_combined", "earn_mdn_1yr_num", 
            "earn_mdn_5yr_num", "acceptance_rate_numeric", 
            "average_net_price_numeric"
        ]
        
        available = sum(1 for metric in metrics if pd.notnull(row[metric]))
        return available / len(metrics)

    # Calculate scores for each dimension
    final_candidates["academic_distance"] = final_candidates.apply(
        lambda x: calculate_academic_distance(x, user_gpa, user_sat), axis=1
    )
    final_candidates["earnings_score"] = final_candidates.apply(calculate_earnings_score, axis=1)
    final_candidates["data_completeness"] = final_candidates.apply(calculate_data_completeness, axis=1)

    # Create composite score with different weights for different scenarios
    def calculate_composite_score(row):
        academic_weight = 0.4
        earnings_weight = 0.4
        completeness_weight = 0.2
        base_score = 0.1  # Ensure all schools get at least some score
        
        scores = [base_score]  # Start with base score
        weights = [0.1]  # Small weight for base score
        
        # Academic distance (inverse it since lower is better)
        if pd.notnull(row["academic_distance"]):
            scores.append(1 - row["academic_distance"])
            weights.append(academic_weight)
            
        # Earnings score
        if pd.notnull(row["earnings_score"]):
            scores.append(row["earnings_score"])
            weights.append(earnings_weight)
            
        # Data completeness
        scores.append(row["data_completeness"])
        weights.append(completeness_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        return sum(s * w for s, w in zip(scores, weights))

    def assign_tier(row):
        # First check if we have enough data for meaningful academic comparison
        has_academic_data = pd.notnull(row["academic_distance"])
        has_earnings = pd.notnull(row["earnings_score"])
        data_score = row["data_completeness"]
        
        # If we have very limited data, just mark as an option
        if data_score < 0.2:  # Less than 20% of key metrics available
            return "Option (Limited Data)"
            
        # If we have academic data, use it for reach/match/safety
        if has_academic_data:
            academic_distance = row["academic_distance"]
            if academic_distance <= 0.2:  # Very close match
                prefix = "Strong"
            elif academic_distance <= 0.4:
                prefix = "Good"
            elif academic_distance <= 0.6:
                prefix = "Potential"
            else:
                prefix = "Consider"
                
            # If we also have earnings data, incorporate that into the tier name
            if has_earnings and row["earnings_score"] > 0.7:
                return f"{prefix} Match (High Earnings)"
            elif has_earnings and row["earnings_score"] > 0.4:
                return f"{prefix} Match (Good Earnings)"
            else:
                return f"{prefix} Match"
        
        # If we only have earnings data
        elif has_earnings:
            if row["earnings_score"] > 0.7:
                return "Option (High Earnings)"
            elif row["earnings_score"] > 0.4:
                return "Option (Good Earnings)"
            else:
                return "Option"
                
        # If we have some data but not enough for academic or earnings comparison
        elif data_score >= 0.2:
            return "Option (Partial Data)"
            
        return "Option"

    # Calculate scores and sort
    final_candidates["composite_score"] = final_candidates.apply(calculate_composite_score, axis=1)
    final_candidates = final_candidates.sort_values(
        by=["composite_score", "data_completeness"],  # Use data completeness as secondary sort
        ascending=[False, False]
    )
    final_candidates["tier"] = final_candidates.apply(assign_tier, axis=1)

    recommended_cols = [
        "name", "average_gpa", "average_sat_composite", "earn_mdn_1yr_num", "earn_mdn_5yr_num",
        "fortune_500_companies", "number_of_students", "acceptance_rate_numeric", "LATITUDE",
        "LONGITUDE", "average_net_price_numeric", "tier", "composite_score"
    ]

    recommendations = final_candidates[recommended_cols].copy()

    recommendations.rename(
        columns={
            "name": "School",
            "average_gpa": "Avg_GPA",
            "average_sat_composite": "Avg_SAT",
            "earn_mdn_1yr_num": "Median_Earnings_1yr",
            "earn_mdn_5yr_num": "Median_Earnings_5yr",
            "fortune_500_companies": "Fortune500_Hirers",
            "number_of_students": "Total_Enrollment",
            "acceptance_rate_numeric": "Admission_Rate",
            "LATITUDE": "Latitude",
            "LONGITUDE": "Longitude",
            "average_net_price_numeric": "Avg_Net_Price",
            "tier": "Recommendation_Tier"
        },
        inplace=True
    )

    recommendations["Has_Salary_Data"] = recommendations["Median_Earnings_1yr"].notna()

    column_order = [
        "School", "Recommendation_Tier", "Has_Salary_Data", "Median_Earnings_1yr", "Median_Earnings_5yr",
        "Avg_GPA", "Avg_SAT", "Fortune500_Hirers", "Total_Enrollment",
        "Admission_Rate", "Avg_Net_Price", "Latitude", "Longitude"
    ]
    recommendations = recommendations[column_order]

    recommendations.reset_index(drop=True, inplace=True)
    return recommendations

if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    if len(sys.argv) != 4:
        print("Usage: python generate_recommendations2.py <gpa> <sat> <program>")
        print("Example: python generate_recommendations2.py 3.6 1300 \"Zoology\"")
        sys.exit(1)
    
    gpa = float(sys.argv[1])
    sat = int(sys.argv[2])
    program = sys.argv[3]
    
    recommendations = recommend_schools(gpa, sat, program)
    
    if not recommendations.empty:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"school_recommendations_{timestamp}.csv"
        
        # Save to CSV
        recommendations.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\nFound {len(recommendations)} schools with {program}")
        print(f"Results saved to: {output_file}")
        
        # Print a preview of the results
        print("\nTop Recommendations:")
        print("=" * 100)
        print(f"{'School':<40} {'Tier':<25} {'Earnings (1yr/5yr)':<20} {'GPA/SAT':<15}")
        print("-" * 100)
        for _, row in recommendations.head(5).iterrows():
            school_name = row['School'][:37] + '...' if len(row['School']) > 37 else row['School']
            earnings = f"${row['Median_Earnings_1yr']:,.0f}/{row['Median_Earnings_5yr']:,.0f}" if pd.notnull(row['Median_Earnings_1yr']) and pd.notnull(row['Median_Earnings_5yr']) else "No data"
            academics = f"{row['Avg_GPA']:.1f}/{row['Avg_SAT']:.0f}" if pd.notnull(row['Avg_GPA']) and pd.notnull(row['Avg_SAT']) else "No data"
            
            print(f"{school_name:<40} {row['Recommendation_Tier']:<25} {earnings:<20} {academics:<15}")
    else:
        print("\nNo recommendations found.")