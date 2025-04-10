import os
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import traceback
# Import the ACT to SAT conversion module
from services.act_sat_conversion import act_to_sat, sat_to_act

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
# ---------------------------------------------------------------------
# Global dictionary + loader for admissions data
# ---------------------------------------------------------------------
admissions_data_dict = {}
def load_admissions_data():
    """
    Load admissions data from 'admission_trends_cleaned.csv'.
    Build a nested dictionary:
        admissions_data_dict[standard_name][year] = { metric: value, ... }
    """
    global admissions_data_dict
    admissions_csv = "recommendation-algo-2/admission_trends_cleaned.csv"
    if not os.path.exists(admissions_csv):
        print(f"Warning: {admissions_csv} not found, skipping admissions data load.")
        return
    try:
        df_ad = pd.read_csv(admissions_csv)
        df_ad["Standard_Name_clean"] = df_ad["Standard_Name"].str.lower().str.strip()
        data_dict = {}
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
# Load admissions data immediately so it's ready
load_admissions_data()
# ---------------------------------------------------------------------
# Helper functions for conversions/parsing
# ---------------------------------------------------------------------
def find_program(program: str) -> Optional[str]:
    """Find the exact program name from the list of valid programs or suggest the closest match."""
    program_clean = program.lower().strip()
    for valid_program in VALID_PROGRAMS:
        if valid_program.lower() == program_clean:
            return valid_program
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
    """Convert a value to numeric, returning NaN if conversion fails."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan
def parse_enrollment(x):
    """Parse enrollment numbers from various formats."""
    if pd.isnull(x):
        return np.nan
    if isinstance(x, (int, float)):
        return x
    x = str(x).strip().replace(',', '')
    try:
        return int(x)
    except ValueError:
        return np.nan
def clean_for_json(data_list):
    """Clean a list of dictionaries for JSON serialization."""
    if not data_list:
        return []
    result = []
    for item in data_list:
        clean_item = {}
        for k, v in item.items():
            if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                clean_item[k] = int(v)
            elif isinstance(v, (np.float64, np.float32, np.float16)):
                clean_item[k] = float(v)
            elif isinstance(v, np.bool_):
                clean_item[k] = bool(v)
            elif pd.isna(v):
                clean_item[k] = None
            else:
                clean_item[k] = v
        result.append(clean_item)
    return result
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
    path_colleges: str = "recommendation-algo-2/colleges_data_cleaned.csv",
    path_programs: str = "recommendation-algo-2/programs_cleaned.csv",
    path_school_sup: str = "recommendation-algo-2/school_sup_data_cleaned.csv",
    path_companies: str = "recommendation-algo-2/companies_data_cleaned.csv",
    verbose: bool = True
):
    """
    Generates a recommendations DataFrame for the given user inputs.
    Merges multiple CSV files, applies transformations, calculates tiers, fallback logic, etc.,
    and attaches both admissions data and selected school support metrics (UGDS, UG, UGDS_WHITE, UGDS_BLACK, UGDS_HISP, UGDS_ASIAN).
    """
    try:
        # Convert ACT to SAT if SAT is not provided but ACT is
        effective_sat = sat
        if (sat is None or sat == 0) and act is not None and act > 0:
            converted_sat = act_to_sat(act)
            if converted_sat is not None:
                effective_sat = converted_sat
                if verbose:
                    print(f"Using converted SAT score {effective_sat} from ACT score {act}")
        
        # 1) Determine the exact program name
        exact_program = find_program(program)
        if exact_program is None:
            if verbose:
                print(f"Could not find a matching program for '{program}'.")
            return pd.DataFrame()
        if verbose:
            print(f"\nLoading data for program: {exact_program}")
        # 2) Load all data files
        if not os.path.exists(path_colleges) or not os.path.exists(path_programs) or not os.path.exists(path_school_sup):
            if verbose:
                print(f"One or more data files not found. Check your file paths!")
            return pd.DataFrame()
        df_colleges = pd.read_csv(path_colleges)
        df_programs = pd.read_csv(path_programs)
        df_school_sup = pd.read_csv(path_school_sup)
        df_colleges["name_clean"] = df_colleges["name"].str.lower().str.strip()
        if "standard_college" in df_programs.columns:
            df_programs["school_name_clean"] = df_programs["standard_college"].str.lower().str.strip()
        else:
            df_programs["school_name_clean"] = df_programs["name"].str.lower().str.strip()
        df_school_sup["school_name_clean"] = df_school_sup["INSTNM"].str.lower().str.strip()
        df_colleges["average_gpa"] = pd.to_numeric(df_colleges["average_gpa"], errors="coerce")
        df_colleges["average_sat_composite"] = pd.to_numeric(df_colleges["average_sat_composite"], errors="coerce")
        # Load companies data if available
        school_to_companies = {}
        if os.path.exists(path_companies):
            df_companies = pd.read_csv(path_companies)
            # Check if required columns exist
            if "company_name" in df_companies.columns and "school" in df_companies.columns:
                for _, row in df_companies.iterrows():
                    company_name = row["company_name"]
                    school_raw = row["school"]
                    if pd.notnull(school_raw):
                        school_clean = str(school_raw).lower().strip()
                        if school_clean not in school_to_companies:
                            school_to_companies[school_clean] = set()
                        school_to_companies[school_clean].add(company_name)
            else:
                if verbose:
                    print(f"Warning: Companies data file missing required columns. Available columns: {df_companies.columns.tolist()}")
        # 3) Filter programs by the exact matched program
        df_prog_filtered = df_programs[df_programs["program"] == exact_program]
        if verbose:
            print(f"\nProgram filtering results:")
            print(f"Programs matching '{exact_program}': {len(df_prog_filtered)}")
        if df_prog_filtered.empty:
            if verbose:
                print(f"No schools found offering program '{exact_program}'.")
            return pd.DataFrame()
        # 4) Merge program data with colleges
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
        # 5) Merge with school_sup (which contains the UGDS fields)
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
        # 6) Combine SAT columns
        def get_sat_value(row):
            if pd.notnull(row["average_sat_composite"]):
                return row["average_sat_composite"]
            return row["SAT_AVG"]
        merged["sat_combined"] = merged.apply(get_sat_value, axis=1)
        # 7) Tier logic based on GPA and SAT differences
        def gpa_tier(row):
            if pd.isnull(row["average_gpa"]):
                return None
            diff = row["average_gpa"] - gpa
            if abs(diff) <= 0.2:
                return "Match"
            elif 0.2 < diff <= 0.4:
                return "Safety"
            elif -0.4 <= diff < -0.2:
                return "Reach"
            else:
                return None
        def sat_tier(row):
            if pd.isnull(row["sat_combined"]):
                return None
            # Use the effective SAT score (either original SAT or converted from ACT)
            diff = row["sat_combined"] - effective_sat
            if abs(diff) <= 100:
                return "Match"
            elif 100 < diff <= 200:
                return "Safety"
            elif -200 <= diff < -100:
                return "Reach"
            else:
                return None
        merged["gpa_tier"] = merged.apply(gpa_tier, axis=1)
        merged["sat_tier"] = merged.apply(sat_tier, axis=1)
        def get_combined_tier(row):
            gpa_val = row["gpa_tier"]
            sat_val = row["sat_tier"]
            if pd.isnull(gpa_val) or pd.isnull(sat_val):
                return None
            if gpa_val == "Match" and sat_val == "Match":
                return "Match"
            elif "Safety" in (gpa_val, sat_val) and "Reach" not in (gpa_val, sat_val):
                return "Safety"
            elif "Reach" in (gpa_val, sat_val) and "Safety" not in (gpa_val, sat_val):
                return "Reach"
            else:
                return None
        merged["tier"] = merged.apply(get_combined_tier, axis=1)
        # 8) Add Fortune 500 companies
        def get_fortune500_companies(row):
            school_clean = row["school_name_clean"]
            if school_clean in school_to_companies:
                return sorted(list(school_to_companies[school_clean]))
            return []
        merged["fortune_500_companies"] = merged.apply(get_fortune500_companies, axis=1)
        # 9) Apply preferences
        def calculate_academic_distance(row, gpa_val, sat_val):
            gpa_distance = 0.0
            sat_distance = 0.0
            count = 0
            if pd.notnull(row["average_gpa"]) and pd.notnull(gpa_val):
                gpa_diff = abs(row["average_gpa"] - gpa_val)
                gpa_distance = gpa_diff / 4.0  # Normalize by max GPA
                count += 1
            if pd.notnull(row["sat_combined"]) and pd.notnull(sat_val):
                sat_diff = abs(row["sat_combined"] - sat_val)
                sat_distance = sat_diff / 1600.0  # Normalize by max SAT
                count += 1
            if count == 0:
                return 1.0  # Maximum distance if no data
            return (gpa_distance + sat_distance) / count
        # 10) Apply preferences to get composite score
        def calculate_composite_score(row):
            score = 0.0
            # Academic match (GPA + SAT)
            if row["tier"] == "Match":
                score += 0.4
            elif row["tier"] == "Safety":
                score += 0.3
            elif row["tier"] == "Reach":
                score += 0.2
            # Location preference - check if LOCALE column exists
            if "LOCALE" in row:
                if location_preference == "Urban" and pd.notnull(row["LOCALE"]) and row["LOCALE"] in [11, 12, 13]:
                    score += 0.1
                elif location_preference == "Suburban" and pd.notnull(row["LOCALE"]) and row["LOCALE"] in [21, 22, 23]:
                    score += 0.1
                elif location_preference == "Rural" and pd.notnull(row["LOCALE"]) and row["LOCALE"] in [31, 32, 33, 41, 42, 43]:
                    score += 0.1
            # Cost preference (lower is better)
            if "average_net_price_numeric" in row and cost_preference == "Low" and pd.notnull(row["average_net_price_numeric"]):
                if row["average_net_price_numeric"] < 20000:
                    score += 0.1
            # Admission rate preference
            if "acceptance_rate_numeric" in row:
                if admission_rate_preference == "High" and pd.notnull(row["acceptance_rate_numeric"]):
                    if row["acceptance_rate_numeric"] > 0.7:
                        score += 0.1
                elif admission_rate_preference == "Medium" and pd.notnull(row["acceptance_rate_numeric"]):
                    if 0.3 <= row["acceptance_rate_numeric"] <= 0.7:
                        score += 0.1
                elif admission_rate_preference == "Low" and pd.notnull(row["acceptance_rate_numeric"]):
                    if row["acceptance_rate_numeric"] < 0.3:
                        score += 0.1
            # Salary preference
            if "earn_mdn_1yr_num" in row and salary_preference == "High" and pd.notnull(row["earn_mdn_1yr_num"]):
                if row["earn_mdn_1yr_num"] > 60000:
                    score += 0.1
            # Fortune 500 preference
            if "fortune_500_companies" in row and fortune500_preference == "Yes" and isinstance(row["fortune_500_companies"], list) and len(row["fortune_500_companies"]) > 0:
                score += 0.1
            return score
        merged["composite_score"] = merged.apply(calculate_composite_score, axis=1)
        # 11) Sort and select top recommendations
        merged = merged.sort_values(by=["composite_score", "tier"], ascending=[False, True])
        # 12) Ensure we have enough recommendations
        min_required = min(number_of_recommendations, 10)  # Cap at 10 for now
        count_final = len(merged[merged["tier"].notna()])
        final_candidates = merged[merged["tier"].notna()].copy()
        if count_final < min_required:
            # Add fallback schools with incomplete data
            fallback_pool = merged[merged["tier"].isna()].copy()
            # Prioritize schools with more complete data
            fallback_pool["data_completeness"] = (
                fallback_pool["average_gpa"].notna().astype(int) +
                fallback_pool["sat_combined"].notna().astype(int) +
                fallback_pool["earn_mdn_1yr_num"].notna().astype(int) +
                fallback_pool["earn_mdn_5yr_num"].notna().astype(int) +
                fallback_pool["acceptance_rate_numeric"].notna().astype(int) +
                fallback_pool["average_net_price_numeric"].notna().astype(int)
            ) / 6.0
            # Alternative method with more granular metrics
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
        # Add the UGDS and related columns from the school_sup data.
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
            "UGDS",
            "UG",
            "UGDS_WHITE",
            "UGDS_BLACK",
            "UGDS_HISP",
            "UGDS_ASIAN",
            "tier"
        ]
        if "composite_score" in final_candidates.columns:
            recommended_cols.append("composite_score")
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
        final_cols = [
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
            "UGDS",
            "UG",
            "UGDS_WHITE",
            "UGDS_BLACK",
            "UGDS_HISP",
            "UGDS_ASIAN",
            "school_name_clean"
        ]
        if "composite_score" in recommendations.columns:
            final_cols.append("composite_score")
        recommendations = recommendations[final_cols]
        recommendations.reset_index(drop=True, inplace=True)
        # Attach admissions data
        recommendations["Admission_Statistics"] = recommendations["school_name_clean"].apply(get_admission_statistics)
        recommendations.drop(columns=["school_name_clean"], inplace=True, errors="ignore")
        return recommendations
    except Exception as e:
        print(f"Error in recommend_schools: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise
