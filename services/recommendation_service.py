import os
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import traceback

# ===================================================================================
# All recommendation logic and helper functions that were previously in app.py
# are now moved to this service file for a more modular, production-friendly design.
# ===================================================================================

VALID_PROGRAMS = []

def load_valid_programs():
    """Attempt to load the list of valid program names from programs_cleaned.csv."""
    try:
        df_programs = pd.read_csv("recommendation-algo-2/programs_cleaned.csv")
        return sorted(df_programs["program"].unique().tolist())
    except Exception as e:
        print(f"Error loading program names: {e}")
        return []

VALID_PROGRAMS = load_valid_programs()

# -----------------------------------------------------
# Global dictionary to store admissions data by school
# -----------------------------------------------------
admissions_data_dict = {}

def load_admissions_data():
    """
    Load the admissions data from 'admission_trends_cleaned.csv'.
    Build a nested dictionary:
        admissions_data_dict[standard_name_clean][year] = {metric: value, metric2: value2, ...}
    """
    global admissions_data_dict
    admissions_csv = "admission_trends_cleaned.csv"
    if not os.path.exists(admissions_csv):
        print(f"Warning: {admissions_csv} not found, skipping admissions data load.")
        return

    try:
        df_ad = pd.read_csv(admissions_csv)
        df_ad["Standard_Name_clean"] = df_ad["Standard_Name"].str.lower().str.strip()
        data_dict = {}

        # Group rows by (standard_name_clean, year) to collect metrics
        grouped = df_ad.groupby(["Standard_Name_clean", "Year"])
        for (std_name_clean, year), group_df in grouped:
            if std_name_clean not in data_dict:
                data_dict[std_name_clean] = {}
            if year not in data_dict[std_name_clean]:
                data_dict[std_name_clean][year] = {}

            for _, row in group_df.iterrows():
                metric = row["Metric"]
                value = row["Value"]
                data_dict[std_name_clean][year][metric] = value

        admissions_data_dict = data_dict
        print("Admissions data loaded into admissions_data_dict.")

    except Exception as ex:
        print(f"Error loading admissions data: {ex}")

def get_admission_statistics(std_name_clean: str):
    """
    Given a school's lowercased name (school_name_clean),
    return a list of { 'year': int, 'metrics': {...} } for each available year.
    """
    if std_name_clean in admissions_data_dict:
        stats_list = []
        for year in sorted(admissions_data_dict[std_name_clean].keys()):
            metrics = admissions_data_dict[std_name_clean][year]
            stats_list.append({
                "year": year,
                "metrics": metrics
            })
        return stats_list
    return []

# Attempt to load admissions data at import time
load_admissions_data()

def find_program(program: str) -> Optional[str]:
    """Find the exact program name from the list of valid programs or suggest the closest match."""
    program_clean = program.lower().strip()
    
    # Try exact match first
    for valid_program in VALID_PROGRAMS:
        if valid_program.lower() == program_clean:
            return valid_program
    
    # If no exact match, compute rough "distance" to find potential suggestion
    if len(VALID_PROGRAMS) == 0:
        return None

    closest_match = None
    min_diff = float('inf')

    for valid_program in VALID_PROGRAMS:
        diff = abs(len(valid_program) - len(program_clean))
        if diff < min_diff:
            min_diff = diff
            closest_match = valid_program
    return closest_match

def numeric_or_nan(x):
    """Attempt to parse x as float, return NaN on failure."""
    try:
        return float(x)
    except:
        return np.nan

def parse_enrollment(x):
    """Parse enrollment numbers that might contain commas or be missing."""
    if pd.isnull(x):
        return None
    if isinstance(x, str):
        x = x.replace(",", "")
    try:
        return int(float(x))
    except:
        return None

def parse_accept_rate(x):
    """Parses acceptance rate if it has a trailing percent sign."""
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
    """Parse strings like '$12,345' into float."""
    if isinstance(x, str):
        x = x.replace("$", "").replace(",", "")
    try:
        return float(x)
    except:
        return np.nan

def recommend_schools(
    program: str,
    gpa: float,
    sat: int,
    act: float,
    location_preference: str,
    cost_preference: str,
    admission_rate_preference: str,
    salary_preference: str,
    fortune500_preference: str,
    number_of_recommendations: int,
    is_random: bool = False,
    path_colleges: str = "recommendation-algo-2/colleges_data_cleaned.csv",
    path_programs: str = "recommendation-algo-2/programs_cleaned.csv",
    path_school_sup: str = "recommendation-algo-2/school_sup_data_cleaned.csv",
    path_companies: str = "recommendation-algo-2/companies_data_cleaned.csv",
    verbose: bool = True
):
    """
    Main function to generate a recommendations DataFrame for the given user inputs.
    The approach merges multiple CSV files, applies transformations, calculates tiers,
    fallback logic, etc., and returns a final DataFrame ready for output.
    """
    try:
        user_program = program
        user_gpa = gpa
        user_sat = sat

        exact_program = find_program(user_program)
        if exact_program is None:
            raise ValueError(f"Invalid program name: '{user_program}'. Please use one of the valid program names.")

        if verbose:
            print(f"\nLoading data for program: {exact_program}")
            print(f"Looking for files in: {os.getcwd()}")
            print("Checking if files exist:")
            print(f"- {path_colleges}: {os.path.exists(path_colleges)}")
            print(f"- {path_programs}: {os.path.exists(path_programs)}")
            print(f"- {path_school_sup}: {os.path.exists(path_school_sup)}")
            print(f"- {path_companies}: {os.path.exists(path_companies)}")

        df_colleges = pd.read_csv(path_colleges)
        df_programs = pd.read_csv(path_programs)
        df_school_sup = pd.read_csv(path_school_sup)
        df_companies = pd.read_csv(path_companies)

        if verbose:
            print("Loaded data files:")
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

        df_prog_filtered = df_programs[df_programs["program"] == exact_program]

        if verbose:
            print(f"\nProgram filtering results:")
            print(f"Programs matching '{exact_program}': {len(df_prog_filtered)}")

        if df_prog_filtered.empty:
            if verbose:
                print(f"No schools found offering program '{exact_program}'.")
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
            print(f"Rows after first merge with colleges: {len(merged)}")
            print(f"Columns now: {merged.columns.tolist()}")

        if "name_col" in merged.columns:
            merged.rename(columns={"name_col": "name"}, inplace=True)

        merged = pd.merge(
            merged,
            df_school_sup,
            on="school_name_clean",
            how="inner",
            suffixes=("_col", "_sup")
        )

        if verbose:
            print(f"Rows after second merge with school_sup: {len(merged)}")
            print(f"Columns now: {merged.columns.tolist()}")

        if "name_col" in merged.columns:
            merged.rename(columns={"name_col": "name"}, inplace=True)
        if "name_prog" in merged.columns:
            merged.rename(columns={"name_prog": "name"}, inplace=True)
        elif "name_x" in merged.columns:
            merged.rename(columns={"name_x": "name"}, inplace=True)
        elif "INSTNM" in merged.columns and "name" not in merged.columns:
            merged.rename(columns={"INSTNM": "name"}, inplace=True)

        if verbose and "name" not in merged.columns:
            print("Could not find a standard 'name' column. Columns are:", merged.columns.tolist())

        if "name" not in merged.columns:
            if verbose:
                print("Error: Could not find 'name' column after merges.")
            return pd.DataFrame()

        def get_sat_value(row):
            if pd.notnull(row["average_sat_composite"]):
                return row["average_sat_composite"]
            return row["SAT_AVG"]

        merged["sat_combined"] = merged.apply(get_sat_value, axis=1)

        # Parse acceptance rate and other numeric values first
        merged["acceptance_rate_numeric"] = merged["acceptance_rate"].apply(parse_accept_rate)
        merged["average_net_price_numeric"] = merged["average_net_price"].apply(parse_dollar_amount)
        merged["earn_mdn_1yr_num"] = merged["earn_mdn_1yr"].apply(numeric_or_nan)
        merged["earn_mdn_5yr_num"] = merged["earn_mdn_5yr"].apply(numeric_or_nan)

        def gpa_tier(row):
            if pd.isnull(row["average_gpa"]):
                return None
            diff = row["average_gpa"] - user_gpa
            if abs(diff) <= 0.1:  # Very close match (within 0.1)
                return "Strong Match"
            elif abs(diff) <= 0.2:  # Close match (within 0.2)
                return "Match"
            elif 0.2 < diff <= 0.4:  # Student's GPA is lower
                return "Safety"
            elif -0.4 <= diff < -0.2:  # Student's GPA is higher
                return "Reach"
            else:
                return None

        def sat_tier(row):
            if pd.isnull(row["sat_combined"]):
                return None
            diff = row["sat_combined"] - user_sat
            if abs(diff) <= 50:  # Very close match (within 50 points)
                return "Strong Match"
            elif abs(diff) <= 100:  # Close match (within 100 points)
                return "Match"
            elif 100 < diff <= 200:  # Student's SAT is lower
                return "Safety"
            elif -200 <= diff < -100:  # Student's SAT is higher
                return "Reach"
            else:
                return None

        def acceptance_rate_tier(row):
            if pd.isnull(row["acceptance_rate_numeric"]):
                return None
            accept_rate = row["acceptance_rate_numeric"]
            # Very low acceptance rates (3-6%) are always considered safety schools
            if accept_rate <= 0.06:  # 6% or lower acceptance rate
                return "Safety"
            elif accept_rate <= 0.15:  # 15% or lower acceptance rate
                return "Reach"
            elif accept_rate >= 0.75:  # 75% or higher acceptance rate
                return "Safety"
            else:
                return "Match"

        def location_tier(row, location_preference):
            if location_preference == "any":
                return "Match"
            
            # If we have location data and a specific preference
            if pd.notnull(row["LATITUDE"]) and pd.notnull(row["LONGITUDE"]):
                # For now, we'll consider schools within the same state as a match
                # This can be enhanced with more sophisticated distance calculations
                if location_preference.lower() in str(row["name"]).lower():
                    return "Match"
                else:
                    return "Reach"
            return None

        merged["gpa_tier"] = merged.apply(gpa_tier, axis=1)
        merged["sat_tier"] = merged.apply(sat_tier, axis=1)
        merged["acceptance_tier"] = merged.apply(acceptance_rate_tier, axis=1)
        merged["location_tier"] = merged.apply(lambda x: location_tier(x, location_preference), axis=1)

        def calculate_academic_distance(row, user_gpa, user_sat):
            """
            Calculate a normalized academic distance between the student and the school.
            Returns a value between 0 (perfect match) and 1 (maximum difference).
            """
            gpa_distance = 0.0
            sat_distance = 0.0
            count = 0
            
            # Calculate GPA distance
            if pd.notnull(row["average_gpa"]) and pd.notnull(user_gpa):
                # Direct difference in GPA points (not normalized)
                gpa_diff = abs(row["average_gpa"] - user_gpa)
                # Convert to a 0-1 scale where 0.3 GPA difference = 1.0
                gpa_distance = min(gpa_diff / 0.3, 1.0)
                count += 1
            
            # Calculate SAT distance
            if pd.notnull(row["sat_combined"]) and pd.notnull(user_sat):
                # Direct difference in SAT points (not normalized)
                sat_diff = abs(row["sat_combined"] - user_sat)
                # Convert to a 0-1 scale where 150 SAT points difference = 1.0
                sat_distance = min(sat_diff / 150, 1.0)
                count += 1
            
            if count == 0:
                return None
            
            # Weight GPA and SAT equally
            return (gpa_distance + sat_distance) / count

        merged["academic_distance"] = merged.apply(
            lambda x: calculate_academic_distance(x, user_gpa, user_sat),
            axis=1
        )

        def assign_tier(row, is_random=False):
            """
            A simplified approach to assigning tiers, balancing academic fit, acceptance rate,
            and differences in GPA/SAT. This avoids overlapping or contradictory rules.

            Steps:
            1) Check for significant academic mismatch first
               - If student is significantly stronger than school's typical profile (distance < 0.20) -> 'Reach'
               - If school is significantly below student's profile (distance > 0.85) -> 'Unrecommended'
            2) If acceptance_rate < 0.10 -> 'Reach' (ultra-selective)
            3) If acceptance_rate > 0.85 -> 'Safety' (very high acceptance)
            4) If 0.20 <= acceptance_rate <= 0.60 -> 'Target' (moderate selectivity)
            5) Else compute academic fit (0.0 is perfect match, up to 1.0 is max difference).
               - If fit > 0.67 -> 'Safety'
               - If fit > 0.33 -> 'Target'
               - Else -> 'Reach'
            """
            # If we have no academic distance data, default to 'Target'
            if pd.isnull(row["academic_distance"]):
                return "Target"

            acceptance_rate = row.get("acceptance_rate_numeric", None)
            distance = row["academic_distance"]  # 0.0 = identical, 1.0 = beyond threshold

            # Step 1: Check for significant academic mismatch first
            # If school is significantly below student's profile
            if distance > 0.85:
                return "Unrecommended"  # Schools significantly below student's profile
            
            # If student is significantly stronger than school's typical profile
            if distance < 0.20:  # Student is significantly stronger than school's typical profile
                # For schools with moderate acceptance rates (20-60%), still consider as Target
                # but with a note that it might be a "Reach" in terms of academic fit
                if acceptance_rate is not None and 0.20 <= acceptance_rate <= 0.60:
                    return "Target (Strong Student)"
                return "Reach"

            # Step 2: Ultra-selective check
            if acceptance_rate is not None and acceptance_rate < 0.10:
                return "Reach"

            # Step 3: Very high acceptance rate check
            if acceptance_rate is not None and acceptance_rate > 0.85:
                return "Safety"

            # Step 4: Moderate acceptance rate check (20-60%)
            if acceptance_rate is not None and 0.20 <= acceptance_rate <= 0.60:
                return "Target"

            # Step 5: Use academic distance for other cases
            if distance > 0.67:
                tier = "Safety"
            elif distance > 0.33:
                tier = "Target"
            else:
                tier = "Reach"

            return tier

        merged["tier"] = merged.apply(assign_tier, axis=1)
        final_candidates = merged[merged["tier"].notnull()].copy()

        # Add attractiveness based on earnings
        def is_attractive(row):
            if pd.isnull(row["earn_mdn_1yr_num"]) and pd.isnull(row["earn_mdn_5yr_num"]):
                return False
            
            # Check if earnings are above average
            earnings_1yr = row["earn_mdn_1yr_num"] if pd.notnull(row["earn_mdn_1yr_num"]) else 0
            earnings_5yr = row["earn_mdn_5yr_num"] if pd.notnull(row["earn_mdn_5yr_num"]) else 0
            
            # Consider attractive if either 1yr or 5yr earnings are above $50,000
            return earnings_1yr > 50000 or earnings_5yr > 50000

        final_candidates["is_attractive"] = final_candidates.apply(is_attractive, axis=1)

        # Update tier names to include attractiveness
        def update_tier_with_attractiveness(row):
            tier = row["tier"]
            if row["is_attractive"]:
                return f"Attractive {tier}"
            return tier

        final_candidates["tier"] = final_candidates.apply(update_tier_with_attractiveness, axis=1)

        final_candidates["acceptance_rate_numeric"] = final_candidates["acceptance_rate"].apply(parse_accept_rate)
        final_candidates["average_net_price_numeric"] = final_candidates["average_net_price"].apply(parse_dollar_amount)
        final_candidates["earn_mdn_1yr_num"] = final_candidates["earn_mdn_1yr"].apply(numeric_or_nan)
        final_candidates["earn_mdn_5yr_num"] = final_candidates["earn_mdn_5yr"].apply(numeric_or_nan)

        def get_companies_for_school(row):
            school_name_clean = row["school_name_clean"]
            return list(school_to_companies.get(school_name_clean, []))

        final_candidates["fortune_500_companies"] = final_candidates.apply(get_companies_for_school, axis=1)

        def calculate_earnings_score(row):
            earnings_1yr = row["earn_mdn_1yr_num"] if pd.notnull(row["earn_mdn_1yr_num"]) else 0
            earnings_5yr = row["earn_mdn_5yr_num"] if pd.notnull(row["earn_mdn_5yr_num"]) else 0
            max_1yr = final_candidates["earn_mdn_1yr_num"].max() or 1
            max_5yr = final_candidates["earn_mdn_5yr_num"].max() or 1
            score_1yr = earnings_1yr / max_1yr if max_1yr > 0 else 0
            score_5yr = earnings_5yr / max_5yr if max_5yr > 0 else 0
            if earnings_1yr > 0 and earnings_5yr > 0:
                return (0.4 * score_1yr + 0.6 * score_5yr)
            elif earnings_1yr > 0:
                return score_1yr * 0.8
            elif earnings_5yr > 0:
                return score_5yr * 0.8
            return None

        def calculate_data_completeness(row):
            metrics = [
                "average_gpa", "sat_combined", "earn_mdn_1yr_num",
                "earn_mdn_5yr_num", "acceptance_rate_numeric", "average_net_price_numeric"
            ]
            available = sum(1 for metric in metrics if pd.notnull(row[metric]))
            return available / len(metrics)

        final_candidates["earnings_score"] = final_candidates.apply(calculate_earnings_score, axis=1)
        final_candidates["data_completeness"] = final_candidates.apply(calculate_data_completeness, axis=1)

        def compute_selectivity_distance(row):
            accept_rate = row["acceptance_rate_numeric"]
            if pd.isnull(accept_rate):
                return 0.0
            penalty = max(0.0, 0.2 - accept_rate) * 10.0
            return penalty

        final_candidates["selectivity_distance"] = final_candidates.apply(compute_selectivity_distance, axis=1)

        def combine_distances(row):
            if pd.notnull(row["academic_distance"]):
                return row["academic_distance"] + row["selectivity_distance"]
            return None

        final_candidates["combined_distance"] = final_candidates.apply(combine_distances, axis=1)

        def calculate_composite_score(row, is_random=False):
            """
            Calculate a composite score that prioritizes academic fit and earnings data.
            """
            base_score = 0.0
            weights = []
            scores = []
            
            # Academic fit (normalized distance)
            if pd.notnull(row["academic_distance"]):
                academic_score = 1 - row["academic_distance"]  # Convert distance to similarity
                scores.append(academic_score)
                weights.append(0.4)  # 40% weight on academic fit
            
            # Earnings data (normalized)
            if pd.notnull(row["earn_mdn_1yr_num"]) or pd.notnull(row["earn_mdn_5yr_num"]):
                earnings_1yr = row["earn_mdn_1yr_num"] if pd.notnull(row["earn_mdn_1yr_num"]) else 0
                earnings_5yr = row["earn_mdn_5yr_num"] if pd.notnull(row["earn_mdn_5yr_num"]) else 0
                max_1yr = final_candidates["earn_mdn_1yr_num"].max() or 1
                max_5yr = final_candidates["earn_mdn_5yr_num"].max() or 1
                
                score_1yr = earnings_1yr / max_1yr if max_1yr > 0 else 0
                score_5yr = earnings_5yr / max_5yr if max_5yr > 0 else 0
                
                if earnings_1yr > 0 and earnings_5yr > 0:
                    earnings_score = (0.4 * score_1yr + 0.6 * score_5yr)
                elif earnings_1yr > 0:
                    earnings_score = score_1yr
                elif earnings_5yr > 0:
                    earnings_score = score_5yr
                else:
                    earnings_score = 0
                
                scores.append(earnings_score)
                weights.append(0.6)  # 60% weight on earnings
            
            # If no scores available, return base score
            if not scores:
                return base_score
            
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Calculate weighted average
            return sum(s * w for s, w in zip(scores, normalized_weights))

        def adjust_composite_score(row):
            if pd.notnull(row["academic_distance"]):
                return row["composite_score"] * (1 - row["academic_distance"] * 0.9)
            return row["composite_score"]

        # Calculate composite scores
        final_candidates["composite_score"] = final_candidates.apply(
            lambda row: calculate_composite_score(row, is_random=is_random), axis=1
        )
        
        # Sort by composite score and data completeness
        final_candidates = final_candidates.sort_values(
            by=["composite_score", "data_completeness"], ascending=[False, False]
        )
        
        # Assign tiers
        final_candidates["tier"] = final_candidates.apply(
            lambda row: assign_tier(row, is_random=is_random), axis=1
        )

        min_required = 10
        count_final = len(final_candidates)
        if count_final < min_required:
            fallback_pool = merged.copy()
            fallback_pool = fallback_pool[~fallback_pool.index.isin(final_candidates.index)]
            fallback_pool = fallback_pool.loc[:, ~fallback_pool.columns.duplicated()].copy()
            fallback_pool.reset_index(drop=True, inplace=True)

            fallback_pool["acceptance_rate_numeric"] = fallback_pool["acceptance_rate"].apply(parse_accept_rate)
            fallback_pool["average_net_price_numeric"] = fallback_pool["average_net_price"].apply(parse_dollar_amount)
            fallback_pool["earn_mdn_1yr_num"] = fallback_pool["earn_mdn_1yr"].apply(numeric_or_nan)
            fallback_pool["earn_mdn_5yr_num"] = fallback_pool["earn_mdn_5yr"].apply(numeric_or_nan)

            def fallback_data_completeness(row):
                metrics = [
                    "average_gpa", "sat_combined", "earn_mdn_1yr_num",
                    "earn_mdn_5yr_num", "acceptance_rate_numeric", "average_net_price_numeric"
                ]
                available = sum(1 for metric in metrics if pd.notnull(row[metric]))
                return available / len(metrics)

            fallback_pool["data_completeness"] = fallback_pool.apply(fallback_data_completeness, axis=1)
            fallback_pool = fallback_pool.sort_values(by="data_completeness", ascending=False)
            fallback_pool.reset_index(drop=True, inplace=True)

            needed = min_required - count_final
            fallback_slice = fallback_pool.head(needed).copy()
            fallback_slice["tier"] = "Fallback: Limited Data"

            if "fortune_500_companies" not in fallback_slice.columns:
                fallback_slice["fortune_500_companies"] = [[] for _ in range(len(fallback_slice))]
            else:
                fallback_slice["fortune_500_companies"] = fallback_slice["fortune_500_companies"].apply(
                    lambda x: x if isinstance(x, list) else []
                )

            final_candidates = final_candidates.loc[:, ~final_candidates.columns.duplicated()].copy()
            final_candidates.reset_index(drop=True, inplace=True)

            missing_columns = [col for col in final_candidates.columns if col not in fallback_slice.columns]
            for col in missing_columns:
                fallback_slice[col] = None
            fallback_slice = fallback_slice[final_candidates.columns]
            final_candidates = pd.concat([final_candidates, fallback_slice], ignore_index=True)

            if verbose:
                print(f"Only found {count_final} strong matches. Added {len(fallback_slice)} fallback schools to reach {min_required}.")

        final_candidates["number_of_students"] = final_candidates["number_of_students"].apply(parse_enrollment)

        # --------------------------------------------------------
        # Include 'school_name_clean' so we can attach admissions
        # --------------------------------------------------------
        recommended_cols = [
            "school_name_clean",
            "name",
            "average_gpa",
            "average_sat_composite",
            "earn_mdn_1yr_num",
            "earn_mdn_5yr_num",
            "fortune_500_companies",
            "number_of_students",
            "acceptance_rate_numeric",
            "LATITUDE",
            "LONGITUDE",
            "average_net_price_numeric",
            "tier",
            "composite_score"
        ]

        if any(col not in final_candidates.columns for col in recommended_cols):
            if verbose:
                print("One or more required columns are missing from final_candidates.")
            return pd.DataFrame()

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
            "School",
            "Recommendation_Tier",
            "Has_Salary_Data",
            "Median_Earnings_1yr",
            "Median_Earnings_5yr",
            "Avg_GPA",
            "Avg_SAT",
            "Fortune500_Hirers",
            "Total_Enrollment",
            "Admission_Rate",
            "Avg_Net_Price",
            "Latitude",
            "Longitude",
            "school_name_clean"
        ]
        recommendations = recommendations[column_order]
        recommendations.reset_index(drop=True, inplace=True)

        # --------------------------------------------------------
        # Attach admissions data to each school
        # --------------------------------------------------------
        recommendations["Admission_Statistics"] = recommendations["school_name_clean"].apply(get_admission_statistics)
        recommendations.drop(columns=["school_name_clean"], inplace=True)

        return recommendations

    except Exception as e:
        print(f"Error in recommend_schools: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise