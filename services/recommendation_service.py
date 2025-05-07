"""
University Recommendation Service

This module provides functionality for generating university recommendations based on various criteria
including academic performance, preferences, and program availability. It processes multiple data sources
to provide personalized school recommendations.

Key Features:
- Program matching and validation
- Academic performance analysis (GPA, SAT/ACT)
- Preference-based scoring (location, cost, etc.)
- Data integration from multiple sources
- Fallback recommendation logic
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Set, Union, Tuple
from datetime import datetime
import traceback
# Import the ACT to SAT conversion module
from services.act_sat_conversion import act_to_sat, sat_to_act

# ===================================================================================
# Program Validation and Data Loading
# ===================================================================================

# Global list of valid program names loaded from CSV
VALID_PROGRAMS: List[str] = []

def load_valid_programs() -> List[str]:
    """
    Load and validate program names from the programs_cleaned.csv file.
    
    Returns:
        List[str]: Sorted list of unique program names, or empty list if loading fails
        
    Raises:
        FileNotFoundError: If the programs CSV file cannot be found
        Exception: For any other errors during file reading or processing
    """
    try:
        # This df_programs is temporary for this function's scope
        df_programs_temp = pd.read_csv("recommendation-algo-2/programs_cleaned.csv")
        return sorted(df_programs_temp["program"].unique().tolist())
    except Exception as e:
        print(f"Error loading program names: {e}")
        return []

VALID_PROGRAMS = load_valid_programs()

# ===================================================================================
# Global DataFrames and Loaders
# ===================================================================================

# Paths (Consider moving to a config file or environment variables later)
PATH_COLLEGES = "recommendation-algo-2/colleges_data_cleaned.csv"
PATH_PROGRAMS = "recommendation-algo-2/programs_cleaned.csv"
PATH_SCHOOL_SUP = "recommendation-algo-2/school_sup_data_cleaned.csv"
PATH_COMPANIES = "recommendation-algo-2/companies_data_cleaned.csv"
PATH_ADMISSIONS_TRENDS = "recommendation-algo-2/admission_trends_cleaned.csv"
PATH_SCHOOL_IMAGES = "recommendation-algo-2/school_plus_image_data_cleaned.csv"

DF_COLLEGES: pd.DataFrame = pd.DataFrame()
DF_PROGRAMS: pd.DataFrame = pd.DataFrame()
DF_SCHOOL_SUP: pd.DataFrame = pd.DataFrame()
DF_COMPANIES: pd.DataFrame = pd.DataFrame()
DF_SCHOOL_IMAGES: pd.DataFrame = pd.DataFrame()
SCHOOL_TO_COMPANIES: Dict[str, Dict[str, int]] = {}
admissions_data_dict: Dict[str, Dict[int, Dict[str, Any]]] = {}

def load_colleges_data() -> pd.DataFrame:
    """Loads and preprocesses colleges data."""
    if not os.path.exists(PATH_COLLEGES):
        print(f"Warning: {PATH_COLLEGES} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(PATH_COLLEGES)
        df["name_clean"] = df["name"].str.lower().str.strip()
        df["average_gpa"] = pd.to_numeric(df["average_gpa"], errors="coerce")
        df["average_sat_composite"] = pd.to_numeric(df["average_sat_composite"], errors="coerce")
        # Add any other one-time transformations here
        print("Colleges data loaded and preprocessed.")
        return df
    except Exception as e:
        print(f"Error loading colleges data: {e}")
        return pd.DataFrame()

def load_programs_data() -> pd.DataFrame:
    """Loads and preprocesses programs data."""
    if not os.path.exists(PATH_PROGRAMS):
        print(f"Warning: {PATH_PROGRAMS} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(PATH_PROGRAMS)
        if "standard_college" in df.columns:
            df["school_name_clean"] = df["standard_college"].str.lower().str.strip()
        else:
            df["school_name_clean"] = df["name"].str.lower().str.strip()
        print("Programs data loaded and preprocessed.")
        return df
    except Exception as e:
        print(f"Error loading programs data: {e}")
        return pd.DataFrame()

def load_school_sup_data() -> pd.DataFrame:
    """Loads and preprocesses school supplementary data."""
    if not os.path.exists(PATH_SCHOOL_SUP):
        print(f"Warning: {PATH_SCHOOL_SUP} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(PATH_SCHOOL_SUP)
        df["school_name_clean"] = df["INSTNM"].str.lower().str.strip()
        # Add any other one-time transformations here (e.g., UGDS parsing if universally applicable)
        print("School supplementary data loaded and preprocessed.")
        return df
    except Exception as e:
        print(f"Error loading school supplementary data: {e}")
        return pd.DataFrame()

def load_companies_data_and_build_map() -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Loads companies data (wide format) and builds the school_to_companies map 
       with alumni counts.
       Map structure: {school_name_clean: {company_name: alumni_count}}
    """ 
    school_to_companies_map: Dict[str, Dict[str, int]] = {}
    path_companies_local = "recommendation-algo-2/companies_data_cleaned.csv"
    
    if not os.path.exists(path_companies_local):
        print(f"Warning: Companies data file not found at {path_companies_local}. SCHOOL_TO_COMPANIES will be empty.")
        return pd.DataFrame(), school_to_companies_map
    try:
        df = pd.read_csv(path_companies_local)
        
        # Identify the maximum number of school_X columns dynamically or assume a fixed max (e.g., 15 from header)
        # For robustness, find all 'school_X' columns
        school_cols = sorted([col for col in df.columns if col.startswith('school_') and col.replace('school_', '').isdigit()])
        max_school_index = 0
        if school_cols:
            max_school_index = max([int(col.replace('school_', '')) for col in school_cols])

        if 'company' not in df.columns:
            print(f"Warning: {path_companies_local} missing required 'company' column.")
            return df, school_to_companies_map
            
        for index, row in df.iterrows():
            company_name = row["company"]
            if pd.isnull(company_name):
                continue # Skip rows with no company name

            for i in range(1, max_school_index + 1): # Iterate from school_1 up to max_school_index
                school_col_name = f"school_{i}"
                alumni_count_col_name = f"alumnicount_{i}"

                if school_col_name in df.columns and pd.notnull(row[school_col_name]):
                    school_raw = row[school_col_name]
                    school_clean_key = str(school_raw).lower().strip()
                    
                    alumni_count = 0 # Default to 0 if count is missing or not a number
                    if alumni_count_col_name in df.columns and pd.notnull(row[alumni_count_col_name]):
                        try:
                            alumni_count = int(row[alumni_count_col_name])
                        except ValueError:
                            if verbose_startup:
                                print(f"Warning: Could not convert alumni count '{row[alumni_count_col_name]}' to int for {company_name} at {school_clean_key}. Using 0.")
                            alumni_count = 0 # Default to 0 if conversion fails
                    
                    if school_clean_key not in school_to_companies_map:
                        school_to_companies_map[school_clean_key] = {}
                    school_to_companies_map[school_clean_key][str(company_name)] = alumni_count
            
        if verbose_startup:
            print(f"Successfully built SCHOOL_TO_COMPANIES map with {len(school_to_companies_map)} school entries.")
            if school_to_companies_map:
                print("Sample school keys in SCHOOL_TO_COMPANIES (first 3):")
                for i, school_key in enumerate(list(school_to_companies_map.keys())[:3]):
                    print(f"  - '{school_key}':")
                    companies_at_school = school_to_companies_map[school_key]
                    for j, company_entry_key in enumerate(list(companies_at_school.keys())[:3]): # Show first 3 companies for this school
                        print(f"    - '{company_entry_key}': {companies_at_school[company_entry_key]} alumni")
                    if len(companies_at_school) > 3: print("      ...")
                if len(school_to_companies_map) > 3: print("  ...")

        return df, school_to_companies_map
    except Exception as e:
        print(f"Error loading companies data from {path_companies_local}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), school_to_companies_map

# Control verbosity of startup messages for SCHOOL_TO_COMPANIES
verbose_startup = True # Set to False to reduce startup log noise once working

def load_admissions_data() -> None:
    """
    Load and process admissions data from admission_trends_cleaned.csv.
    
    The data is structured as a nested dictionary:
    admissions_data_dict[standard_name][year] = { metric: value, ... }
    
    This function:
    1. Reads the admissions CSV file
    2. Cleans and standardizes school names
    3. Groups data by school and year
    4. Stores the processed data in the global admissions_data_dict
    
    Raises:
        FileNotFoundError: If the admissions CSV file cannot be found
        Exception: For any other errors during data processing
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
        
        # Group data by school name and year
        grouped = df_ad.groupby(["Standard_Name_clean", "Year"])
        for (std_name_clean, year), group_df in grouped:
            if std_name_clean not in data_dict:
                data_dict[std_name_clean] = {}
            if year not in data_dict[std_name_clean]:
                data_dict[std_name_clean][year] = {}
                
            # Store metrics for each year
            for _, row in group_df.iterrows():
                metric = row["Metric"]
                value = row["Value"]
                data_dict[std_name_clean][year][metric] = value
                
        admissions_data_dict = data_dict
        print("Admissions data loaded into admissions_data_dict.")
    except Exception as ex:
        print(f"Error loading admissions data: {ex}")

def get_admission_statistics(std_name_clean: str) -> List[Dict[str, Any]]:
    """
    Retrieve admission statistics for a given school.
    
    Args:
        std_name_clean (str): Lowercased and stripped school name
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing yearly admission statistics,
                             or empty list if no data found
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

def load_school_images_data() -> pd.DataFrame:
    """Loads and preprocesses school images data."""
    if not os.path.exists(PATH_SCHOOL_IMAGES):
        print(f"Warning: {PATH_SCHOOL_IMAGES} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(PATH_SCHOOL_IMAGES)
        # Create school_name_clean from instnm for joining
        if "instnm" in df.columns:
            df["school_name_clean"] = df["instnm"].astype(str).str.lower().str.strip()
        else:
            print(f"Warning: 'instnm' column not found in {PATH_SCHOOL_IMAGES}. Cannot create 'school_name_clean'.")
            # Return an empty DataFrame or a df without school_name_clean if critical downstream
            return pd.DataFrame() 
        
        # Select only necessary columns to keep the DataFrame lean
        # Ensure target columns exist before selection
        required_cols = ["school_name_clean", "cdn_url", "thumb_link"]
        # Add unitid as well, it might be useful for more robust joining later
        if "unitid" in df.columns: required_cols.append("unitid") 
            
        for col in required_cols:
            if col not in df.columns and col not in ["unitid"]: # unitid is optional for now in this selection
                print(f"Warning: Expected column '{col}' not found in {PATH_SCHOOL_IMAGES}.")
                # Depending on how critical these are, you might return an empty df
        
        # Filter to columns that actually exist to prevent KeyErrors
        existing_required_cols = [col for col in required_cols if col in df.columns]
        df_selected = df[existing_required_cols].copy() # Use .copy() if further modifications are planned on df_selected

        print("School images data loaded and preprocessed.")
        return df_selected
    except Exception as e:
        print(f"Error loading school images data: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

# Load admissions data on module import
load_admissions_data()

# Load other datasets on module import
DF_COLLEGES = load_colleges_data()
DF_PROGRAMS = load_programs_data()
DF_SCHOOL_SUP = load_school_sup_data()
DF_COMPANIES, SCHOOL_TO_COMPANIES = load_companies_data_and_build_map()
DF_SCHOOL_IMAGES = load_school_images_data()

# ===================================================================================
# Constants for Scoring & Logic (Example - can be expanded)
# ===================================================================================
GPA_MATCH_THRESHOLD_STRICT = 0.2
GPA_MATCH_THRESHOLD_LOOSE = 0.4
SAT_MATCH_THRESHOLD_STRICT = 100
SAT_MATCH_THRESHOLD_LOOSE = 200

# ===================================================================================
# Data Processing and Helper Functions
# ===================================================================================

def _merge_dataframes(
    df_prog_filtered: pd.DataFrame,
    df_colleges: pd.DataFrame,
    df_school_sup: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """Merges the core dataframes based on cleaned school names."""
    if verbose:
        print(f"_merge_dataframes: Initial program-filtered rows: {len(df_prog_filtered)}")

    merged_df = pd.merge(
        df_prog_filtered,
        df_colleges,
        left_on="school_name_clean",
        right_on="name_clean",
        how="inner",
        suffixes=("_prog", "_coll") # Changed suffix to avoid conflict with potential 'col' column
    )
    if verbose:
        print(f"_merge_dataframes: Rows after first merge (programs with colleges): {len(merged_df)}")

    if merged_df.empty:
        return pd.DataFrame() 

    # Prioritize name from colleges (df_colleges originally has 'name')
    # If 'name_coll' (from colleges) exists, rename it to 'name'.
    # Else if 'name_prog' (from programs) exists and 'name' is not already there, use 'name_prog'.
    if "name_coll" in merged_df.columns:
        merged_df.rename(columns={"name_coll": "name"}, inplace=True)
        if "name_prog" in merged_df.columns and "name_prog" != "name": # drop a duplicated name_prog if name_coll was chosen
            merged_df.drop(columns=["name_prog"], inplace=True, errors='ignore')
    elif "name_prog" in merged_df.columns and "name" not in merged_df.columns:
        merged_df.rename(columns={"name_prog": "name"}, inplace=True)
    
    # Before the second merge, ensure essential columns from df_colleges that might be used later
    # are present in merged_df, possibly with suffixes if there were clashes avoided by pandas.
    # For example, if df_colleges had 'average_gpa', it might be 'average_gpa_coll' in merged_df.
    # We need to ensure they are standardized or explicitly selected later.
    # For now, we rely on the suffixes and select the correct one later or ensure they don't clash.

    merged_df = pd.merge(
        merged_df,
        df_school_sup, # df_school_sup has 'INSTNM' as its original name column
        on="school_name_clean",
        how="inner",
        suffixes=("", "_sup") # _left refers to merged_df from previous step, _sup to df_school_sup
    )

    if verbose:
        # Corrected single print for the second merge result
        print(f"_merge_dataframes: Rows after second merge (with school_sup): {len(merged_df)}")
        # print(f"_merge_dataframes: Columns after all merges: {merged_df.columns.tolist()}") # Optional: very verbose

    if merged_df.empty:
        return pd.DataFrame()

    # Final name column standardization: 
    # If 'name' (from first merge) doesn't exist or was suffixed, try to use INSTNM_sup from school_sup data.
    if "name" not in merged_df.columns and "INSTNM_sup" in merged_df.columns:
        merged_df.rename(columns={"INSTNM_sup": "name"}, inplace=True)
    elif "name_left" in merged_df.columns and "name" not in merged_df.columns: # If 'name' from first merge got suffixed
        merged_df.rename(columns={"name_left": "name"}, inplace=True)
    
    # Ensure other critical columns that might have gotten suffixes are standardized or handled.
    # Example: If df_colleges had 'average_gpa' and df_school_sup also had something similar,
    # we need to decide which one to use or how to coalesce them.
    # For now, assume the loader functions for DF_COLLEGES and DF_SCHOOL_SUP provide the primary versions
    # and we rely on selecting the non-suffixed or correctly suffixed ones later if pandas created them.
    # This part might need more sophisticated column coalescing if clashes are common for key data points.

    if verbose and "name" not in merged_df.columns:
        print("Warning: _merge_dataframes: Could not find a standard 'name' column after all merges.")
    
    return merged_df

def _calculate_academic_metrics(
    df: pd.DataFrame, 
    gpa: float, 
    effective_sat: int, 
    school_to_companies: Dict[str, Set[str]]
) -> pd.DataFrame:
    """Calculates GPA tier, SAT tier, combined tier, and Fortune 500 companies."""
    if df.empty:
        return df

    # Ensure necessary columns from merges are present
    if "average_gpa" not in df.columns:
        df["average_gpa"] = np.nan # Add if missing to prevent key errors
    if "average_sat_composite" not in df.columns and "SAT_AVG" not in df.columns:
        df["average_sat_composite"] = np.nan # Add if missing
        df["SAT_AVG"] = np.nan

    def get_sat_value(row: pd.Series) -> Union[float, np.nan]:
        if pd.notnull(row.get("average_sat_composite")):
            return row["average_sat_composite"]
        return row.get("SAT_AVG") # .get avoids KeyError if SAT_AVG also missing

    def gpa_tier(row: pd.Series) -> Optional[str]:
        if pd.isnull(row.get("average_gpa")):
            return None
        diff = row["average_gpa"] - gpa
        if abs(diff) <= GPA_MATCH_THRESHOLD_STRICT:
            return "Match"
        elif GPA_MATCH_THRESHOLD_STRICT < diff <= GPA_MATCH_THRESHOLD_LOOSE: # Student GPA is lower
            return "Safety" # School is safety for student
        elif -GPA_MATCH_THRESHOLD_LOOSE <= diff < -GPA_MATCH_THRESHOLD_STRICT: # Student GPA is higher
            return "Reach" # School is reach for student
        # Simplified: if diff > GPA_MATCH_THRESHOLD_LOOSE (school GPA much higher), it's a Safety
        # if diff < -GPA_MATCH_THRESHOLD_LOOSE (school GPA much lower), it's a Reach (less likely for student)
        # This logic might need review based on desired tier definitions
        # The original logic was: Safety if 0.2 < diff <= 0.4; Reach if -0.4 <= diff < -0.2
        # Let's stick to the original logic for now.
        if GPA_MATCH_THRESHOLD_STRICT < diff <= GPA_MATCH_THRESHOLD_LOOSE:
            return "Safety" 
        elif -GPA_MATCH_THRESHOLD_LOOSE <= diff < -GPA_MATCH_THRESHOLD_STRICT:
            return "Reach"  
        return None # Or "Far Reach" / "Far Safety" if more categories are needed

    def sat_tier(row: pd.Series) -> Optional[str]:
        if pd.isnull(row.get("sat_combined")):
            return None
        diff = row["sat_combined"] - effective_sat
        if abs(diff) <= SAT_MATCH_THRESHOLD_STRICT:
            return "Match"
        elif SAT_MATCH_THRESHOLD_STRICT < diff <= SAT_MATCH_THRESHOLD_LOOSE: # School SAT is higher
            return "Safety" # School is safety for student
        elif -SAT_MATCH_THRESHOLD_LOOSE <= diff < -SAT_MATCH_THRESHOLD_STRICT: # School SAT is lower
            return "Reach" # School is reach for student
        return None

    def get_combined_tier(row: pd.Series) -> Optional[str]:
        gpa_val = row.get("gpa_tier")
        sat_val = row.get("sat_tier")
        # If either is missing, we can't determine a combined tier confidently
        if pd.isnull(gpa_val) or pd.isnull(sat_val):
             # If one is present, could default to that one, or require both.
             # Current logic requires both, effectively.
            return None 
        if gpa_val == "Match" and sat_val == "Match":
            return "Match"
        # Prioritize Reach if either is Reach
        if gpa_val == "Reach" or sat_val == "Reach":
            return "Reach"
        # Then prioritize Safety if either is Safety (and no Reach)
        if gpa_val == "Safety" or sat_val == "Safety":
            return "Safety"
        # Fallback if tiers are mixed in an unhandled way (e.g. one match, one other)
        # or if only one score was available and tiered.
        # Based on original: if "Safety" in (gpa_val, sat_val) and "Reach" not in (gpa_val, sat_val) -> Safety
        # Based on original: if "Reach" in (gpa_val, sat_val) and "Safety" not in (gpa_val, sat_val) -> Reach
        # The logic above simplifies: Match > Reach > Safety. 
        # Let's refine to match original more closely:
        if (gpa_val == "Match" and sat_val == "Safety") or \
           (gpa_val == "Safety" and sat_val == "Match") or \
           (gpa_val == "Safety" and sat_val == "Safety"):
            return "Safety"
        if (gpa_val == "Match" and sat_val == "Reach") or \
           (gpa_val == "Reach" and sat_val == "Match") or \
           (gpa_val == "Reach" and sat_val == "Reach"):
            return "Reach"
        return None # Should ideally be covered by above, or indicates only one component had data

    def get_fortune500_companies(row: pd.Series) -> List[str]:
        school_clean = row.get("school_name_clean") # Use .get for safety
        if school_clean and school_clean in school_to_companies:
            return sorted(list(school_to_companies[school_clean]))
        return []

    df_calc = df.copy()
    df_calc["sat_combined"] = df_calc.apply(get_sat_value, axis=1)
    df_calc["gpa_tier"] = df_calc.apply(gpa_tier, axis=1)
    df_calc["sat_tier"] = df_calc.apply(sat_tier, axis=1)
    df_calc["tier"] = df_calc.apply(get_combined_tier, axis=1)
    df_calc["fortune_500_companies"] = df_calc.apply(get_fortune500_companies, axis=1)
    
    return df_calc

def find_program(program: str) -> Optional[str]:
    """
    Find the exact program name from the list of valid programs or suggest the closest match.
    
    Args:
        program (str): Program name to search for
        
    Returns:
        Optional[str]: Exact match if found, closest match if no exact match,
                      or None if no valid programs exist
    """
    program_clean = program.lower().strip()
    
    # First try exact match
    for valid_program in VALID_PROGRAMS:
        if valid_program.lower() == program_clean:
            return valid_program
            
    if len(VALID_PROGRAMS) == 0:
        return None
        
    # If no exact match, find closest by length
    closest_match = None
    min_diff = float('inf')
    for valid_program in VALID_PROGRAMS:
        diff = abs(len(valid_program) - len(program_clean))
        if diff < min_diff:
            min_diff = diff
            closest_match = valid_program
    return closest_match

def numeric_or_nan(x: Any) -> Union[float, np.nan]:
    """
    Safely convert a value to numeric, returning NaN if conversion fails.
    
    Args:
        x (Any): Value to convert to numeric
        
    Returns:
        Union[float, np.nan]: Converted numeric value or NaN if conversion fails
    """
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def parse_enrollment(x: Any) -> Union[int, np.nan]:
    """
    Parse enrollment numbers from various formats.
    
    Args:
        x (Any): Enrollment value to parse
        
    Returns:
        Union[int, np.nan]: Parsed enrollment number or NaN if parsing fails
    """
    if pd.isnull(x):
        return np.nan
    if isinstance(x, (int, float)):
        return x
    x = str(x).strip().replace(',', '')
    try:
        return int(x)
    except ValueError:
        return np.nan

def parse_percentage_rate(x: Any) -> Union[float, np.nan]:
    """
    Safely convert a value (potentially with '%') to a numeric rate (decimal),
    returning NaN if conversion fails.
    Handles cases like '3%', '0.03', 3, 0.03.
    """
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).strip()
        if s.endswith('%'):
            # Remove % and convert to float, then divide by 100
            return float(s[:-1]) / 100.0
        else:
            # Assume it's already a decimal rate or needs conversion
            val = float(s)
            # Optional: Add a check if the value seems like a whole percentage (e.g., 3 instead of 0.03)
            # This might be needed if data format is inconsistent
            # if val > 1 and val <= 100: 
            #    print(f"Warning: Interpreting rate {val} as percentage, dividing by 100.")
            #    return val / 100.0
            return val
    except (ValueError, TypeError):
        return np.nan

def clean_for_json(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean a list of dictionaries for JSON serialization by converting numpy types to Python native types.
    
    Args:
        data_list (List[Dict[str, Any]]): List of dictionaries to clean
        
    Returns:
        List[Dict[str, Any]]: Cleaned list of dictionaries with native Python types
    """
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

# ===================================================================================
# Main Recommendation Logic
# ===================================================================================

def recommend_schools(
    program: str,
    gpa: float,
    sat: int,
    act: float,
    number_of_recommendations: int,
    location_preference: Optional[str] = "any",
    cost_preference: Optional[str] = "any",
    admission_rate_preference: Optional[str] = "any",
    salary_preference: Optional[str] = "any",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Simplified university recommendation function.
    Focuses on academic matching (GPA, SAT) and direct data merging.
    Complex preference scoring and fallback logic are temporarily removed for clarity.
    """
    try:
        if verbose:
            print(f"--- Simplified recommend_schools called for program: {program} ---")
            print(f"Global DataFrames loaded - Colleges: {not DF_COLLEGES.empty}, Programs: {not DF_PROGRAMS.empty}, School_Sup: {not DF_SCHOOL_SUP.empty}")

        # 1. Input Processing
        effective_sat = sat
        if (sat is None or sat == 0) and act is not None and act > 0:
            converted_sat = act_to_sat(act)
            if converted_sat is not None:
                effective_sat = converted_sat
                if verbose: print(f"Using converted SAT score {effective_sat} from ACT score {act}")

        exact_program = find_program(program)
        if verbose: print(f"Found exact_program: {exact_program}")
        if exact_program is None:
            if verbose: print(f"Program '{program}' not found in VALID_PROGRAMS.")
            return pd.DataFrame()

        # 2. Filter and Merge Data
        df_prog_filtered = DF_PROGRAMS[DF_PROGRAMS["program"] == exact_program].copy()
        if df_prog_filtered.empty:
            if verbose: print(f"No schools found offering program '{exact_program}' in DF_PROGRAMS.")
            return pd.DataFrame()
        if verbose: print(f"Found {len(df_prog_filtered)} school listings for program '{exact_program}'.")

        # Merge with Colleges data
        # DF_COLLEGES has 'name_clean', 'average_gpa', 'average_sat_composite', 'name', 'ADM_RATE' (for acceptance_rate_numeric)
        # 'earn_mdn_1yr', 'earn_mdn_5yr', 'LATITUDE', 'LONGITUDE', 'average_net_price'
        merged_df = pd.merge(
            df_prog_filtered,
            DF_COLLEGES,
            left_on="school_name_clean", # from DF_PROGRAMS
            right_on="name_clean",       # from DF_COLLEGES
            how="inner",
            suffixes=("_prog", "_coll")
        )
        if merged_df.empty:
            if verbose: print(f"No matching schools after merging programs with colleges.")
            return pd.DataFrame()
        if verbose: print(f"After merging with colleges: {len(merged_df)} rows.")
        
        # Determine school_display_name with fallbacks using combine_first for prioritized filling
        # Priority: 1. DF_PROGRAMS's "name" (becomes name_prog after first merge with _prog suffix)
        #           2. DF_COLLEGES's "name" (becomes name_coll after first merge with _coll suffix)
        #           3. DF_SCHOOL_SUP's "INSTNM" (could be INSTNM_sup or INSTNM after second merge)
        #           4. school_name_clean (the join key, as a last resort)

        # Initialize with the highest priority: name_prog from DF_PROGRAMS
        if "name_prog" in merged_df.columns:
            merged_df["school_display_name"] = merged_df["name_prog"].copy()
        else:
            # If name_prog isn't present (should be, if DF_PROGRAMS has "name"), start with NaNs
            merged_df["school_display_name"] = pd.Series([np.nan] * len(merged_df), index=merged_df.index, dtype=object)
        
        # Fallback to name_coll from DF_COLLEGES
        if "name_coll" in merged_df.columns:
            merged_df["school_display_name"] = merged_df["school_display_name"].combine_first(merged_df["name_coll"])

        # Fallback to INSTNM from DF_SCHOOL_SUP
        # The second merge (with DF_SCHOOL_SUP) uses suffixes=("", "_sup").
        # So, if INSTNM from DF_SCHOOL_SUP conflicted with an existing column, it became INSTNM_sup.
        # Otherwise, it remained INSTNM.
        if "INSTNM_sup" in merged_df.columns: # Prefer suffixed version if it exists
            merged_df["school_display_name"] = merged_df["school_display_name"].combine_first(merged_df["INSTNM_sup"])
        elif "INSTNM" in merged_df.columns: # Check for non-suffixed INSTNM (likely from DF_SCHOOL_SUP)
            merged_df["school_display_name"] = merged_df["school_display_name"].combine_first(merged_df["INSTNM"])
        
        # Final fallback to school_name_clean
        if "school_name_clean" in merged_df.columns:
            merged_df["school_display_name"] = merged_df["school_display_name"].combine_first(merged_df["school_name_clean"])
        
        # Log if any display names are still NaN after all fallbacks
        if merged_df["school_display_name"].isna().any():
            count_na = merged_df["school_display_name"].isna().sum()
            if verbose: print(f"Warning: {count_na} school_display_name entries are still NaN after all fallbacks.")

        # Merge with School Supplementary data
        # DF_SCHOOL_SUP has 'school_name_clean', 'INSTNM', 'UGDS', 'UG', 'UGDS_WHITE', 'UGDS_BLACK', 'UGDS_HISP', 'UGDS_ASIAN'
        # It also might have 'LATITUDE', 'LONGITUDE', 'ADM_RATE' if DF_COLLEGES doesn't
        merged_df = pd.merge(
            merged_df,
            DF_SCHOOL_SUP,
            on="school_name_clean",
            how="inner",
            suffixes=("", "_sup") # Let _sup apply to clashing columns from DF_SCHOOL_SUP
        )
        if merged_df.empty:
            if verbose: print(f"No matching schools after merging with school_sup data.")
            return pd.DataFrame()
        if verbose: print(f"After merging with school_sup: {len(merged_df)} rows.")
        if verbose: print(f"Columns after all merges: {merged_df.columns.tolist()}")

        # 3. Academic Tier Calculation
        # Ensure 'average_gpa' and 'average_sat_composite' (from DF_COLLEGES) are present
        merged_df["Avg_GPA_School"] = pd.to_numeric(merged_df.get("average_gpa"), errors="coerce")
        merged_df["Avg_SAT_School"] = pd.to_numeric(merged_df.get("average_sat_composite"), errors="coerce")
        
        # Fallback for SAT if 'average_sat_composite' is null but 'SAT_AVG' (from school_sup) is present
        if "SAT_AVG_sup" in merged_df.columns: # Check if SAT_AVG_sup exists
             merged_df["Avg_SAT_School"] = merged_df["Avg_SAT_School"].fillna(pd.to_numeric(merged_df["SAT_AVG_sup"], errors="coerce"))
        elif "SAT_AVG" in merged_df.columns: # Check for non-suffixed SAT_AVG from colleges
             merged_df["Avg_SAT_School"] = merged_df["Avg_SAT_School"].fillna(pd.to_numeric(merged_df["SAT_AVG"], errors="coerce"))


        def calculate_gpa_tier(school_gpa, student_gpa):
            if pd.isnull(school_gpa) or student_gpa is None: return "Unknown"
            diff = school_gpa - student_gpa
            if abs(diff) <= GPA_MATCH_THRESHOLD_STRICT: return "Match"
            if GPA_MATCH_THRESHOLD_STRICT < diff <= GPA_MATCH_THRESHOLD_LOOSE: return "Safety"
            if -GPA_MATCH_THRESHOLD_LOOSE <= diff < -GPA_MATCH_THRESHOLD_STRICT: return "Reach"
            if diff > GPA_MATCH_THRESHOLD_LOOSE : return "Far Safety" # School GPA much higher
            if diff < -GPA_MATCH_THRESHOLD_LOOSE: return "Far Reach" # School GPA much lower
            return "Unknown"

        def calculate_sat_tier(school_sat, student_sat):
            if pd.isnull(school_sat) or student_sat is None: return "Unknown"
            diff = school_sat - student_sat
            if abs(diff) <= SAT_MATCH_THRESHOLD_STRICT:
                return "Match"
            elif SAT_MATCH_THRESHOLD_STRICT < diff <= SAT_MATCH_THRESHOLD_LOOSE: return "Safety"
            elif -SAT_MATCH_THRESHOLD_LOOSE <= diff < -SAT_MATCH_THRESHOLD_STRICT: return "Reach"
            if diff > SAT_MATCH_THRESHOLD_LOOSE : return "Far Safety"
            if diff < -SAT_MATCH_THRESHOLD_LOOSE: return "Far Reach"
            return "Unknown"

        merged_df["gpa_tier_calc"] = merged_df.apply(lambda row: calculate_gpa_tier(row["Avg_GPA_School"], gpa), axis=1)
        merged_df["sat_tier_calc"] = merged_df.apply(lambda row: calculate_sat_tier(row["Avg_SAT_School"], effective_sat), axis=1)

        # Simplified overall tier - prioritize Match, then Reach, then Safety
        def determine_overall_tier(gpa_t, sat_t):
            if gpa_t == "Match" and sat_t == "Match": return "Strong Match"
            if gpa_t == "Match" or sat_t == "Match": return "Match"
            if gpa_t == "Reach" or sat_t == "Reach": return "Reach"
            if gpa_t == "Safety" or sat_t == "Safety": return "Safety"
            if gpa_t == "Far Reach" or sat_t == "Far Reach": return "Far Reach"
            if gpa_t == "Far Safety" or sat_t == "Far Safety": return "Far Safety"
            return "Limited Academic Data"
        
        merged_df["Recommendation_Tier"] = merged_df.apply(lambda row: determine_overall_tier(row["gpa_tier_calc"], row["sat_tier_calc"]), axis=1)

        # 4. Prepare Output DataFrame - ensuring all Pydantic model fields
        recommendations = pd.DataFrame()
        
        recommendations["School"] = merged_df["school_display_name"]
        recommendations["Recommendation_Tier"] = merged_df["Recommendation_Tier"]
        
        # Salary Data
        merged_df["earn_mdn_1yr_num"] = pd.to_numeric(merged_df.get("earn_mdn_1yr"), errors="coerce") # From DF_COLLEGES
        merged_df["earn_mdn_5yr_num"] = pd.to_numeric(merged_df.get("earn_mdn_5yr"), errors="coerce") # From DF_COLLEGES
        recommendations["Has_Salary_Data"] = merged_df["earn_mdn_1yr_num"].notna() | merged_df["earn_mdn_5yr_num"].notna()
        recommendations["Median_Earnings_1yr"] = merged_df["earn_mdn_1yr_num"]
        recommendations["Median_Earnings_5yr"] = merged_df["earn_mdn_5yr_num"]
        
        recommendations["Avg_GPA"] = merged_df["Avg_GPA_School"]
        recommendations["Avg_SAT"] = merged_df["Avg_SAT_School"]
        
        # Fortune 500 Hirers (as List[Dict[str, Any]] to include counts)
        def get_f500_hirer_details(sc_name_clean_lookup):
            companies_with_counts = SCHOOL_TO_COMPANIES.get(sc_name_clean_lookup, {})
            hirer_list = []
            for company, count in companies_with_counts.items():
                hirer_list.append({"company_name": company, "alumni_count": count})
            # Sort by company name for consistent output, or by count if preferred
            return sorted(hirer_list, key=lambda x: x["company_name"])
        
        if 'school_name_clean' in merged_df.columns:
            recommendations["Fortune500_Hirers"] = merged_df["school_name_clean"].apply(get_f500_hirer_details)
            if verbose:
                # Check if any hirer data (beyond just empty lists of dicts) was found
                total_hirer_objects = recommendations["Fortune500_Hirers"].apply(len).sum()
                if total_hirer_objects == 0 and len(SCHOOL_TO_COMPANIES) > 0:
                    print("Debug: Fortune500_Hirers (list of dicts) is all empty, but SCHOOL_TO_COMPANIES is populated.")
                    print(f"  Sample school_name_clean values from merged_df (first 3 used for lookup): {merged_df['school_name_clean'].unique()[:3]}")
        else:
            print("Warning: 'school_name_clean' column not found in merged_df for Fortune500 lookup.")
            recommendations["Fortune500_Hirers"] = [[] for _ in range(len(merged_df))] # Series of empty lists

        # Enrollment - prefer 'number_of_students' from DF_COLLEGES, fallback to 'UGDS' or 'UGDS_sup'
        if 'number_of_students' in merged_df.columns:
            recommendations["Total_Enrollment"] = merged_df['number_of_students'].apply(parse_enrollment)
        elif 'UGDS_sup' in merged_df.columns:
             recommendations["Total_Enrollment"] = merged_df['UGDS_sup'].apply(parse_enrollment)
        elif 'UGDS' in merged_df.columns: # Non-suffixed from first merge if it exists
            recommendations["Total_Enrollment"] = merged_df['UGDS'].apply(parse_enrollment)
        else:
            recommendations["Total_Enrollment"] = np.nan

        # Admission Rate - prefer 'ADM_RATE' from DF_COLLEGES, fallback to 'ADM_RATE_sup'
        # Ensure it's numeric. The Pydantic model expects float.
        adm_rate_series = pd.Series(np.nan, index=merged_df.index)
        if 'ADM_RATE' in merged_df.columns: # From DF_COLLEGES
            adm_rate_series = pd.to_numeric(merged_df['ADM_RATE'], errors='coerce')
        if 'ADM_RATE_sup' in merged_df.columns: # From DF_SCHOOL_SUP
            adm_rate_series = adm_rate_series.fillna(pd.to_numeric(merged_df['ADM_RATE_sup'], errors='coerce'))
        recommendations["Admission_Rate"] = adm_rate_series

        # Avg Net Price from DF_COLLEGES
        recommendations["Avg_Net_Price"] = pd.to_numeric(merged_df.get("average_net_price"), errors="coerce")
        
        # Latitude / Longitude - prefer from DF_COLLEGES, then DF_SCHOOL_SUP
        lat_series = pd.Series(np.nan, index=merged_df.index)
        if 'LATITUDE' in merged_df.columns: lat_series = pd.to_numeric(merged_df['LATITUDE'], errors='coerce')
        if 'LATITUDE_sup' in merged_df.columns: lat_series = lat_series.fillna(pd.to_numeric(merged_df['LATITUDE_sup'], errors='coerce'))
        recommendations["Latitude"] = lat_series

        lon_series = pd.Series(np.nan, index=merged_df.index)
        if 'LONGITUDE' in merged_df.columns: lon_series = pd.to_numeric(merged_df['LONGITUDE'], errors='coerce')
        if 'LONGITUDE_sup' in merged_df.columns: lon_series = lon_series.fillna(pd.to_numeric(merged_df['LONGITUDE_sup'], errors='coerce'))
        recommendations["Longitude"] = lon_series

        # Admission Statistics (using 'school_name_clean' which should be unique key)
        recommendations["Admission_Statistics"] = merged_df["school_name_clean"].apply(get_admission_statistics)

        # Enrollment Demographics (from DF_SCHOOL_SUP, so likely suffixed if not unique)
        # Using .get for safety in case columns don't exist after merge
        ugds_main_col = 'UGDS_sup' if 'UGDS_sup' in merged_df.columns else 'UGDS'
        recommendations["Undergraduate_Enrollment"] = pd.to_numeric(merged_df.get(ugds_main_col), errors="coerce")
        
        ugds_white_col = 'UGDS_WHITE_sup' if 'UGDS_WHITE_sup' in merged_df.columns else 'UGDS_WHITE'
        recommendations["White_Enrollment_Percent"] = pd.to_numeric(merged_df.get(ugds_white_col), errors="coerce")
        
        ugds_black_col = 'UGDS_BLACK_sup' if 'UGDS_BLACK_sup' in merged_df.columns else 'UGDS_BLACK'
        recommendations["Black_Enrollment_Percent"] = pd.to_numeric(merged_df.get(ugds_black_col), errors="coerce")
        
        ugds_hisp_col = 'UGDS_HISP_sup' if 'UGDS_HISP_sup' in merged_df.columns else 'UGDS_HISP'
        recommendations["Hispanic_Enrollment_Percent"] = pd.to_numeric(merged_df.get(ugds_hisp_col), errors="coerce")

        ugds_asian_col = 'UGDS_ASIAN_sup' if 'UGDS_ASIAN_sup' in merged_df.columns else 'UGDS_ASIAN'
        recommendations["Asian_Enrollment_Percent"] = pd.to_numeric(merged_df.get(ugds_asian_col), errors="coerce")

        # 5. Sort and Limit
        # Simple sort for now: by tier (custom sort order) then by a proxy for quality/fit if available
        tier_order = ["Strong Match", "Match", "Reach", "Safety", "Far Reach", "Far Safety", "Limited Academic Data", "Unknown"]
        recommendations["Recommendation_Tier"] = pd.Categorical(recommendations["Recommendation_Tier"], categories=tier_order, ordered=True)
        
        # Example secondary sort: higher salary, lower admission rate (more selective)
        # Handle NaNs in sort keys appropriately (e.g., na_position='last')
        recommendations.sort_values(
            by=["Recommendation_Tier", "Median_Earnings_1yr", "Admission_Rate"],
            ascending=[True, False, True], # Tier asc, Salary desc, Adm_Rate asc
            na_position='last',
            inplace=True
        )
        
        final_recommendations = recommendations.head(number_of_recommendations)
        
        if verbose: print(f"Returning {len(final_recommendations)} recommendations.")
        if verbose and final_recommendations.empty:
            print(f"No recommendations generated for program {exact_program} with current criteria.")
            print(f"Merged DF had {len(merged_df)} rows before final selection.")

        return final_recommendations

    except Exception as e:
        print(f"Error in simplified recommend_schools: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # It's often better to let app.py handle the HTTPException
        # but if we want to provide a specific message from here:
        # raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        return pd.DataFrame() # Or raise the exception

def get_school_statistics(school_name_query: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Retrieves and computes various statistics for a given school name.

    Args:
        school_name_query (str): The name of the school to query (can be display or standardized form).
        verbose (bool): If True, prints debug information during execution.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing school statistics if found, otherwise None.
    """
    school_name_clean_found: Optional[str] = None
    school_display_name_found: Optional[str] = None
    data_sources: List[str] = []

    query_lower_strip = str(school_name_query).lower().strip()

    # Initializing to prevent UnboundLocalError in edge cases
    college_row_data = pd.Series(dtype=object)
    sup_row_data = pd.Series(dtype=object)

    # Strategy: Find school_name_clean and best display_name
    # 1. Check DF_COLLEGES (best source for display name and clean name)
    if not DF_COLLEGES.empty:
        matched_by_clean = DF_COLLEGES[DF_COLLEGES["name_clean"] == query_lower_strip]
        if len(matched_by_clean) == 1:
            school_name_clean_found = matched_by_clean.iloc[0]["name_clean"]
            school_display_name_found = matched_by_clean.iloc[0]["name"]
            college_row_data = matched_by_clean.iloc[0]
            if verbose: print(f"Found in DF_COLLEGES by clean name: {school_display_name_found}")
            data_sources.append("DF_COLLEGES")
        elif not matched_by_clean.empty:
            if verbose: print(f"Ambiguous clean name match in DF_COLLEGES for '{query_lower_strip}'. Count: {len(matched_by_clean)}")
        else:
            if 'name' in DF_COLLEGES.columns:
                temp_lower_name_series = DF_COLLEGES["name"].astype(str).str.lower().str.strip()
                matched_by_display = DF_COLLEGES[temp_lower_name_series == query_lower_strip]
                if len(matched_by_display) == 1:
                    school_name_clean_found = matched_by_display.iloc[0]["name_clean"]
                    school_display_name_found = matched_by_display.iloc[0]["name"]
                    college_row_data = matched_by_display.iloc[0]
                    if verbose: print(f"Found in DF_COLLEGES by display name: {school_display_name_found}")
                    data_sources.append("DF_COLLEGES")
                elif not matched_by_display.empty:
                     if verbose: print(f"Ambiguous display name match in DF_COLLEGES for '{query_lower_strip}'. Count: {len(matched_by_display)}")

    # 2. If not found in DF_COLLEGES, check DF_PROGRAMS (primarily to find school_name_clean)
    if school_name_clean_found is None and not DF_PROGRAMS.empty:
        df_programs_name_col_exists = 'name' in DF_PROGRAMS.columns
        matched_prog_by_clean = DF_PROGRAMS[DF_PROGRAMS["school_name_clean"] == query_lower_strip]
        if not matched_prog_by_clean.empty:
            school_name_clean_found = matched_prog_by_clean.iloc[0]["school_name_clean"]
            school_display_name_found = matched_prog_by_clean.iloc[0]["name"] if df_programs_name_col_exists else school_name_clean_found
            if verbose: print(f"Found school_name_clean '{school_name_clean_found}' via DF_PROGRAMS by clean name. Initial display: {school_display_name_found}")
            data_sources.append("DF_PROGRAMS")
        elif df_programs_name_col_exists:
            temp_lower_prog_name_series = DF_PROGRAMS["name"].astype(str).str.lower().str.strip()
            matched_prog_by_display = DF_PROGRAMS[temp_lower_prog_name_series == query_lower_strip]
            if not matched_prog_by_display.empty:
                school_name_clean_found = matched_prog_by_display.iloc[0]["school_name_clean"]
                school_display_name_found = matched_prog_by_display.iloc[0]["name"]
                if verbose: print(f"Found school_name_clean '{school_name_clean_found}' via DF_PROGRAMS by display name: {school_display_name_found}")
                data_sources.append("DF_PROGRAMS")
        
        if school_name_clean_found and "DF_COLLEGES" not in data_sources and not DF_COLLEGES.empty:
            college_ref_match = DF_COLLEGES[DF_COLLEGES["name_clean"] == school_name_clean_found]
            if not college_ref_match.empty:
                school_display_name_found = college_ref_match.iloc[0]["name"]
                college_row_data = college_ref_match.iloc[0]
                if verbose: print(f"Upgraded display name to '{school_display_name_found}' from DF_COLLEGES.")
                data_sources.append("DF_COLLEGES")
            
    # 3. If still not found, check DF_SCHOOL_SUP (INSTNM for display, school_name_clean for key)
    if school_name_clean_found is None and not DF_SCHOOL_SUP.empty:
        matched_sup_by_clean = DF_SCHOOL_SUP[DF_SCHOOL_SUP["school_name_clean"] == query_lower_strip]
        if not matched_sup_by_clean.empty:
            school_name_clean_found = matched_sup_by_clean.iloc[0]["school_name_clean"]
            school_display_name_found = matched_sup_by_clean.iloc[0].get("INSTNM", school_name_clean_found)
            sup_row_data = matched_sup_by_clean.iloc[0]
            if verbose: print(f"Found school_name_clean '{school_name_clean_found}' via DF_SCHOOL_SUP by clean name. Initial display: {school_display_name_found}")
            data_sources.append("DF_SCHOOL_SUP")
        else:
            if 'INSTNM' in DF_SCHOOL_SUP.columns:
                temp_lower_instnm_series = DF_SCHOOL_SUP["INSTNM"].astype(str).str.lower().str.strip()
                matched_sup_by_display = DF_SCHOOL_SUP[temp_lower_instnm_series == query_lower_strip]
                if not matched_sup_by_display.empty:
                    school_name_clean_found = matched_sup_by_display.iloc[0]["school_name_clean"]
                    school_display_name_found = matched_sup_by_display.iloc[0]["INSTNM"]
                    sup_row_data = matched_sup_by_display.iloc[0]
                    if verbose: print(f"Found school_name_clean '{school_name_clean_found}' via DF_SCHOOL_SUP by display name: {school_display_name_found}")
                    data_sources.append("DF_SCHOOL_SUP")

        if school_name_clean_found and "DF_COLLEGES" not in data_sources and not DF_COLLEGES.empty:
            college_ref_match = DF_COLLEGES[DF_COLLEGES["name_clean"] == school_name_clean_found]
            if not college_ref_match.empty:
                school_display_name_found = college_ref_match.iloc[0]["name"]
                college_row_data = college_ref_match.iloc[0]
                if verbose: print(f"Upgraded display name to '{school_display_name_found}' from DF_COLLEGES.")
                data_sources.append("DF_COLLEGES")
            elif school_name_clean_found and "DF_COLLEGES" in data_sources and college_row_data.empty:
                college_ref_match = DF_COLLEGES[DF_COLLEGES["name_clean"] == school_name_clean_found]
                if not college_ref_match.empty: college_row_data = college_ref_match.iloc[0]

        if school_name_clean_found and "DF_SCHOOL_SUP" in data_sources and sup_row_data.empty:
             sup_data_match = DF_SCHOOL_SUP[DF_SCHOOL_SUP["school_name_clean"] == school_name_clean_found]
             if not sup_data_match.empty: sup_row_data = sup_data_match.iloc[0]


    if not school_name_clean_found:
        if verbose: print(f"School '{school_name_query}' not found after checking all sources.")
        return None

    school_display_name_found = school_display_name_found if school_display_name_found else school_name_clean_found

    stats: Dict[str, Any] = {
        "school_name_display": school_display_name_found,
        "school_name_standardized": school_name_clean_found,
        "data_sources_used": sorted(list(set(data_sources))) 
    }

    # --- Image URLs (from DF_SCHOOL_IMAGES) ---
    stats["image_cdn_url"] = None
    stats["image_thumbnail_url"] = None
    if not DF_SCHOOL_IMAGES.empty and school_name_clean_found:
        image_row = DF_SCHOOL_IMAGES[DF_SCHOOL_IMAGES["school_name_clean"] == school_name_clean_found]
        if not image_row.empty:
            # .get with default None for safety if columns are somehow missing despite loader checks
            stats["image_cdn_url"] = image_row.iloc[0].get("cdn_url")
            stats["image_thumbnail_url"] = image_row.iloc[0].get("thumb_link")
            if "DF_SCHOOL_IMAGES" not in stats["data_sources_used"]:
                stats["data_sources_used"].append("DF_SCHOOL_IMAGES")

    if not DF_PROGRAMS.empty:
        school_programs_df = DF_PROGRAMS[DF_PROGRAMS["school_name_clean"] == school_name_clean_found].copy()
        if not school_programs_df.empty:
            school_programs_df["earn_mdn_1yr_num"] = pd.to_numeric(school_programs_df["earn_mdn_1yr"], errors='coerce')
            school_programs_df["earn_mdn_5yr_num"] = pd.to_numeric(school_programs_df["earn_mdn_5yr"], errors='coerce')
            
            stats["avg_program_median_earnings_1yr"] = school_programs_df["earn_mdn_1yr_num"].mean(skipna=True)
            stats["avg_program_median_earnings_5yr"] = school_programs_df["earn_mdn_5yr_num"].mean(skipna=True)
            stats["programs_offered_count"] = school_programs_df["program"].nunique()
            
            program_salary_details_list: List[Dict[str, Any]] = []
            for _, row in school_programs_df.iterrows():
                program_salary_details_list.append({
                    "program_name": row["program"],
                    "median_earnings_1yr": row["earn_mdn_1yr_num"] if pd.notna(row["earn_mdn_1yr_num"]) else None,
                    "median_earnings_5yr": row["earn_mdn_5yr_num"] if pd.notna(row["earn_mdn_5yr_num"]) else None,
                })
            stats["program_salary_details"] = sorted(program_salary_details_list, key=lambda x: str(x.get("program_name")))
            if "DF_PROGRAMS" not in stats["data_sources_used"]: stats["data_sources_used"].append("DF_PROGRAMS")
        else:
            stats["avg_program_median_earnings_1yr"] = None
            stats["avg_program_median_earnings_5yr"] = None
            stats["programs_offered_count"] = 0
            stats["program_salary_details"] = []
    else:
        stats["avg_program_median_earnings_1yr"] = None
        stats["avg_program_median_earnings_5yr"] = None
        stats["programs_offered_count"] = 0
        stats["program_salary_details"] = []

    # --- Gather general school stats from DF_COLLEGES (using college_row_data) ---
    if not college_row_data.empty:
        stats["average_gpa"] = numeric_or_nan(college_row_data.get("average_gpa"))
        stats["average_sat"] = numeric_or_nan(college_row_data.get("average_sat_composite"))
        # Admission rate: DF_COLLEGES might have 'acceptance_rate' or 'ADM_RATE'
        adm_rate_coll = parse_percentage_rate(college_row_data.get("acceptance_rate")) # USE NEW PARSER
        if pd.isna(adm_rate_coll):
             adm_rate_coll = parse_percentage_rate(college_row_data.get("ADM_RATE")) # USE NEW PARSER
        stats["admission_rate"] = adm_rate_coll
        stats["total_enrollment"] = parse_enrollment(college_row_data.get("number_of_students"))
        stats["average_net_price"] = numeric_or_nan(college_row_data.get("average_net_price"))
        stats["latitude"] = numeric_or_nan(college_row_data.get("LATITUDE"))
        stats["longitude"] = numeric_or_nan(college_row_data.get("LONGITUDE"))
        if "DF_COLLEGES" not in stats["data_sources_used"]: stats["data_sources_used"].append("DF_COLLEGES")

    # --- Gather stats from DF_SCHOOL_SUP (using sup_row_data, and fallback for some general stats) ---
    if not sup_row_data.empty:
        # Fill in missing general stats if not already populated by DF_COLLEGES
        if pd.isna(stats.get("admission_rate")) : stats["admission_rate"] = parse_percentage_rate(sup_row_data.get("ADM_RATE")) # USE NEW PARSER
        if pd.isna(stats.get("average_sat")): stats["average_sat"] = numeric_or_nan(sup_row_data.get("SAT_AVG"))
        if pd.isna(stats.get("total_enrollment")): stats["total_enrollment"] = parse_enrollment(sup_row_data.get("UGDS"))
        if pd.isna(stats.get("latitude")): stats["latitude"] = numeric_or_nan(sup_row_data.get("LATITUDE"))
        if pd.isna(stats.get("longitude")): stats["longitude"] = numeric_or_nan(sup_row_data.get("LONGITUDE"))
        
        stats["undergraduate_enrollment"] = parse_enrollment(sup_row_data.get("UGDS"))
        stats["white_enrollment_percent"] = numeric_or_nan(sup_row_data.get("UGDS_WHITE"))
        stats["black_enrollment_percent"] = numeric_or_nan(sup_row_data.get("UGDS_BLACK"))
        stats["hispanic_enrollment_percent"] = numeric_or_nan(sup_row_data.get("UGDS_HISP"))
        stats["asian_enrollment_percent"] = numeric_or_nan(sup_row_data.get("UGDS_ASIAN"))
        if "DF_SCHOOL_SUP" not in stats["data_sources_used"]: stats["data_sources_used"].append("DF_SCHOOL_SUP")

    stats["admission_statistics"] = get_admission_statistics(school_name_clean_found)
    if stats["admission_statistics"] and "Admissions Data" not in stats["data_sources_used"]:
         stats["data_sources_used"].append("Admissions Data")

    hirer_details_list: List[Dict[str, Any]] = []
    if school_name_clean_found in SCHOOL_TO_COMPANIES:
        for company, count in SCHOOL_TO_COMPANIES[school_name_clean_found].items():
            hirer_details_list.append({"company_name": str(company), "alumni_count": int(count)})
        hirer_details_list = sorted(hirer_details_list, key=lambda x: x["company_name"])
        if "DF_COMPANIES" not in stats["data_sources_used"]: stats["data_sources_used"].append("DF_COMPANIES")
    stats["fortune_500_hirers"] = hirer_details_list
    
    # Ensure all keys from Pydantic model are present, filling with None if missing
    expected_keys = [
        "average_gpa", "average_sat", "admission_rate", "total_enrollment", 
        "undergraduate_enrollment", "average_net_price", "latitude", "longitude",
        "white_enrollment_percent", "black_enrollment_percent", 
        "hispanic_enrollment_percent", "asian_enrollment_percent",
        "avg_program_median_earnings_1yr", "avg_program_median_earnings_5yr",
        "programs_offered_count", "program_salary_details", "admission_statistics",
        "fortune_500_hirers",
        "image_cdn_url", "image_thumbnail_url" # Added new keys
    ]
    for key in expected_keys:
        if key not in stats:
            if key == "program_salary_details" or key == "fortune_500_hirers":
                stats[key] = []
            elif key == "programs_offered_count":
                stats[key] = 0
            else:
                stats[key] = None

    # Final sort for data_sources_used
    stats["data_sources_used"] = sorted(list(set(stats["data_sources_used"])))

    if verbose: print(f"Successfully compiled statistics for {school_display_name_found} from sources: {stats['data_sources_used']}")
    return stats

# Make sure to add get_school_statistics to __all__ if you have one, or ensure it's importable.
