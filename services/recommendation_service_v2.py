"""
University Recommendation Service - Version 2

This module provides an advanced recommendation system based on multi-factor scoring
and user personalization, as outlined in plan_v2_recommendations.md.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple

# Attempt to import thefuzz for optional fuzzy matching, but don't make it a hard dependency
try:
    from thefuzz import process as fuzz_process
    THEFUZZ_AVAILABLE = True
except ImportError:
    THEFUZZ_AVAILABLE = False
    print("Warning: 'thefuzz' library not found. Fuzzy program matching will not be available.")

# Global DataFrames - these will be populated by initialize_v2_data
DF_COLLEGES_V2: pd.DataFrame = pd.DataFrame()
DF_PROGRAMS_V2: pd.DataFrame = pd.DataFrame()
DF_SCHOOL_SUP_V2: pd.DataFrame = pd.DataFrame()
DF_SCHOOL_IMAGES_V2: pd.DataFrame = pd.DataFrame()
SCHOOL_TO_COMPANIES_V2: Dict[str, Dict[str, int]] = {}
ADMISSIONS_DATA_DICT_V2: Dict[str, Dict[int, Dict[str, Any]]] = {}

# --- Mappings and Constants ---
STATE_TO_REGION_MAP: Dict[str, str] = {
    'AL': 'Southeast', 'AK': 'West', 'AZ': 'Southwest', 'AR': 'South', 'CA': 'West',
    'CO': 'West', 'CT': 'Northeast', 'DE': 'Northeast', 'FL': 'Southeast', 'GA': 'Southeast',
    'HI': 'West', 'ID': 'West', 'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest',
    'KS': 'Midwest', 'KY': 'South', 'LA': 'South', 'ME': 'Northeast', 'MD': 'Northeast',
    'MA': 'Northeast', 'MI': 'Midwest', 'MN': 'Midwest', 'MS': 'South', 'MO': 'Midwest',
    'MT': 'West', 'NE': 'Midwest', 'NV': 'West', 'NH': 'Northeast', 'NJ': 'Northeast',
    'NM': 'Southwest', 'NY': 'Northeast', 'NC': 'Southeast', 'ND': 'Midwest', 'OH': 'Midwest',
    'OK': 'South', 'OR': 'West', 'PA': 'Northeast', 'RI': 'Northeast', 'SC': 'Southeast',
    'SD': 'Midwest', 'TN': 'South', 'TX': 'Southwest', 'UT': 'West', 'VT': 'Northeast',
    'VA': 'Southeast', 'WA': 'West', 'WV': 'South', 'WI': 'Midwest', 'WY': 'West',
    # Add US territories if present in data
    'AS': 'Territories', 'GU': 'Territories', 'MP': 'Territories', 'PR': 'Territories', 'VI': 'Territories'
}

SCHOOL_SIZE_THRESHOLDS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "small": (0, 1999),
    "medium": (2000, 14999),
    "large": (15000, float('inf'))
}

# --- Helper Functions ---

def initialize_v2_data(df_colleges, df_programs, df_school_sup, school_to_companies, admissions_data, df_school_images, state_region_map=STATE_TO_REGION_MAP):
    """Initializes global DataFrames for V2 service. Called at app startup."""
    global DF_COLLEGES_V2, DF_PROGRAMS_V2, DF_SCHOOL_SUP_V2, SCHOOL_TO_COMPANIES_V2, ADMISSIONS_DATA_DICT_V2, DF_SCHOOL_IMAGES_V2, STATE_TO_REGION_MAP
    DF_COLLEGES_V2 = df_colleges.copy() if df_colleges is not None and not df_colleges.empty else pd.DataFrame()
    DF_PROGRAMS_V2 = df_programs.copy() if df_programs is not None and not df_programs.empty else pd.DataFrame()
    DF_SCHOOL_SUP_V2 = df_school_sup.copy() if df_school_sup is not None and not df_school_sup.empty else pd.DataFrame()
    SCHOOL_TO_COMPANIES_V2 = school_to_companies.copy() if school_to_companies is not None else {}
    ADMISSIONS_DATA_DICT_V2 = admissions_data.copy() if admissions_data is not None else {}
    DF_SCHOOL_IMAGES_V2 = df_school_images.copy() if df_school_images is not None and not df_school_images.empty else pd.DataFrame()
    STATE_TO_REGION_MAP = state_region_map # Use provided or default
    print("V2 Recommendation Service Data Initialized.")
    # Basic checks
    if DF_COLLEGES_V2.empty: print("Warning: DF_COLLEGES_V2 is empty in V2 service after initialization.")
    if DF_PROGRAMS_V2.empty: print("Warning: DF_PROGRAMS_V2 is empty in V2 service after initialization.")

def _normalize_value(value: Optional[float], min_val: Optional[float], max_val: Optional[float], smaller_is_better: bool = False) -> float:
    """Helper to normalize a single value given min and max."""
    if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return 0.5 # Neutral score
    if smaller_is_better:
        return float(np.clip((max_val - value) / (max_val - min_val), 0.0, 1.0))
    else:
        return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))

# --- Sub-Score Calculation Functions ---

def calculate_academic_fit_score(student_gpa: float, student_sat: Optional[int], 
                                 school_avg_gpa: Optional[float], school_avg_sat: Optional[int], 
                                 academic_focus: str) -> float:
    if pd.isna(school_avg_gpa) and pd.isna(school_avg_sat):
        return 0.1 # Low score if no school academic data

    gpa_fit_score = 0.5 # Neutral default
    if not pd.isna(school_avg_gpa) and student_gpa is not None:
        gpa_diff = school_avg_gpa - student_gpa # Positive: school GPA > student GPA (harder for student)
        if academic_focus == "match":
            gpa_fit_score = max(0.0, 1.0 - abs(gpa_diff) / 0.5) 
        elif academic_focus == "challenge_me": 
            if gpa_diff >= -0.1: # School GPA is similar or higher
                gpa_fit_score = 0.5 + (gpa_diff / 0.8) * 0.5 # Max 1.0 if school is 0.8+ higher (broader range for challenge)
            else: # School GPA is significantly lower
                gpa_fit_score = 0.5 + (gpa_diff / 1.0) * 0.5 # Penalize more if school is much easier
        elif academic_focus == "less_stress": 
            if gpa_diff <= 0.1: # School GPA is similar or lower
                gpa_fit_score = 0.5 - (gpa_diff / 0.8) * 0.5 # Max 1.0 if school is 0.8+ lower
            else: # School GPA is significantly higher
                gpa_fit_score = 0.5 - (gpa_diff / 1.0) * 0.5 
    
    sat_fit_score = 0.5 # Neutral default
    if not pd.isna(school_avg_sat) and student_sat is not None:
        sat_diff = school_avg_sat - student_sat # Positive: school SAT > student SAT
        if academic_focus == "match":
            sat_fit_score = max(0.0, 1.0 - abs(sat_diff) / 200.0)
        elif academic_focus == "challenge_me":
            if sat_diff >= -50: # School SAT is similar or higher
                sat_fit_score = 0.5 + (sat_diff / 300.0) * 0.5 
            else:
                sat_fit_score = 0.5 + (sat_diff / 400.0) * 0.5
        elif academic_focus == "less_stress":
            if sat_diff <= 50: # School SAT is similar or lower
                sat_fit_score = 0.5 - (sat_diff / 300.0) * 0.5
            else:
                sat_fit_score = 0.5 - (sat_diff / 400.0) * 0.5

    # Combine scores: if one is missing, use the other; otherwise average
    if pd.isna(school_avg_gpa) or student_gpa is None:
        final_score = sat_fit_score
    elif pd.isna(school_avg_sat) or student_sat is None:
        final_score = gpa_fit_score
    else:
        final_score = (gpa_fit_score * 0.6) + (sat_fit_score * 0.4) # Slightly more weight to GPA
    
    return float(np.clip(final_score, 0.0, 1.0))


def calculate_program_outcome_score(program_earn_1yr: Optional[float], program_earn_5yr: Optional[float],
                                    norm_stats_1yr: Dict[str, float], norm_stats_5yr: Dict[str, float]) -> float:
    score_1yr = _normalize_value(program_earn_1yr, norm_stats_1yr.get('min'), norm_stats_1yr.get('max'))
    score_5yr = _normalize_value(program_earn_5yr, norm_stats_5yr.get('min'), norm_stats_5yr.get('max'))

    if pd.notna(program_earn_1yr) and pd.notna(program_earn_5yr):
        final_score = (score_1yr * 0.4) + (score_5yr * 0.6)
    elif pd.notna(score_5yr) and score_5yr != 0.5 : # Check not neutral from missing data
        final_score = score_5yr
    elif pd.notna(score_1yr) and score_1yr != 0.5:
        final_score = score_1yr
    else:
        final_score = 0.1 # Low score if no salary data or only neutral scores from normalization
    return float(np.clip(final_score, 0.0, 1.0))

def calculate_affordability_score(school_net_price: Optional[float], student_max_net_price: Optional[int],
                                  norm_stats_price: Dict[str, float]) -> float:
    if pd.isna(school_net_price):
        return 0.3 # Neutral score if price unknown

    if student_max_net_price is not None:
        if school_net_price <= student_max_net_price:
            # Score is 1 if free, 0.33 at max_net_price, linearly in between
            return max(0.0, 1.0 - (school_net_price / (student_max_net_price * 1.5)) if student_max_net_price > 0 else 1.0)
        else:
            return 0.0 # Exceeds student's max budget
    else:
        # No student budget, use global normalization (lower is better)
        return _normalize_value(school_net_price, norm_stats_price.get('min'), norm_stats_price.get('max'), smaller_is_better=True)

def calculate_location_score(school_state: Optional[str], school_city: Optional[str], # school_city for future use with region maps
                             preferred_states: Optional[List[str]], preferred_region: Optional[str]) -> float:
    score = 0.0
    state_match = False
    region_match = False

    # Determine school_region using STATE_TO_REGION_MAP
    school_region_from_map = STATE_TO_REGION_MAP.get(school_state) if school_state else None

    if preferred_states and school_state and school_state in preferred_states:
        state_match = True
    
    if preferred_region and school_region_from_map and school_region_from_map.lower() == preferred_region.lower():
        region_match = True

    if preferred_states and preferred_region:
        if state_match and region_match: score = 1.0      # Perfect match on both
        elif state_match: score = 0.7                      # State matches, region doesn't
        elif region_match: score = 0.6                    # Region matches, state doesn't (less likely to be preferred but possible)
    elif preferred_states:
        if state_match: score = 1.0
    elif preferred_region:
        if region_match: score = 1.0
    else: 
        return 1.0 # Max score if no location preference specified
    
    return score

def calculate_selectivity_score(school_acceptance_rate: Optional[float], selectivity_preference: str) -> float:
    if pd.isna(school_acceptance_rate):
        return 0.5 # Neutral if unknown

    if selectivity_preference == "any":
        return 1.0 # Not a factor for this user, so max contribution
    
    # Ensure rate is 0-1
    rate = school_acceptance_rate
    if rate > 1.0 and rate <=100.0: # Handles if rate is in percent form
        rate = rate / 100.0
    rate = np.clip(rate, 0.0, 1.0)

    if selectivity_preference == "moderate":
        # Gaussian-like score, peak at 0.4 (40% acceptance), tapers off
        return float(np.clip(np.exp(-((rate - 0.4)**2) / (2 * 0.15**2)), 0,1))
    elif selectivity_preference == "high_challenge":
        # Lower acceptance rate is better (score = 1 - rate)
        return float(np.clip(1.0 - rate, 0,1))
    return 0.5 # Default for unhandled preference

def get_school_size_category(num_students: Optional[float]) -> Optional[str]:
    if pd.isna(num_students):
        return None
    if num_students < SCHOOL_SIZE_THRESHOLDS["small"][1] + 1: return "small"
    if num_students < SCHOOL_SIZE_THRESHOLDS["medium"][1] + 1: return "medium"
    return "large"

def calculate_environment_score(school_type: Optional[str], school_num_students: Optional[float],
                                preferred_school_type: Optional[str], preferred_school_sizes: Optional[List[str]]) -> float:
    type_score = 0.5
    size_score = 0.5
    prefs_counted = 0

    school_size_category = get_school_size_category(school_num_students)

    if preferred_school_type and preferred_school_type != "any":
        prefs_counted += 1
        if school_type and school_type.lower() == preferred_school_type.lower():
            type_score = 1.0
        else:
            type_score = 0.0
    else:
        type_score = 1.0 # No preference means it's a match

    if preferred_school_sizes:
        prefs_counted += 1
        if school_size_category and school_size_category in preferred_school_sizes:
            size_score = 1.0
        else:
            size_score = 0.0
    else:
        size_score = 1.0 # No preference means it's a match
        
    if prefs_counted == 0: return 1.0 # No env preferences, perfect match
    return float(np.clip((type_score + size_score) / prefs_counted if prefs_counted > 0 else 1.0 , 0, 1))

def calculate_career_opportunities_score(school_name_clean: str, norm_stats_hirers: Dict[str, float]) -> float:
    if not SCHOOL_TO_COMPANIES_V2 or school_name_clean not in SCHOOL_TO_COMPANIES_V2:
        return 0.1 # Low score if no data for this school
    
    num_companies_this_school = len(SCHOOL_TO_COMPANIES_V2[school_name_clean])
    return _normalize_value(float(num_companies_this_school), norm_stats_hirers.get('min'), norm_stats_hirers.get('max'))

def determine_composite_weights(base_weights: Dict[str, float], preferences: Dict) -> Dict[str, float]:
    adj_weights = base_weights.copy()
    pref_cost_importance = preferences.get("cost", {}).get("importance", "medium")
    pref_career_importance = preferences.get("career_outcomes_importance", "medium")

    weight_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5} # Define multipliers

    adj_weights["affordability"] *= weight_multipliers.get(pref_cost_importance, 1.0)
    
    career_multiplier = weight_multipliers.get(pref_career_importance, 1.0)
    adj_weights["program_outcome"] *= career_multiplier
    adj_weights["career_ops"] *= career_multiplier
    
    if preferences.get("location", {}).get("states") or preferences.get("location", {}).get("region"):
        adj_weights["location"] *= 1.5
    else: # No specific location preference, slightly reduce its default impact
        adj_weights["location"] *= 0.7

    if preferences.get("academic_focus") == "challenge_me" and preferences.get("selectivity_preference") != "any":
        adj_weights["selectivity"] *= 1.2
    if preferences.get("selectivity_preference") == "high_challenge":
         adj_weights["selectivity"] *= 1.3

    total_weight = sum(adj_weights.values())
    if total_weight == 0 or pd.isna(total_weight): return {k: 1.0/len(base_weights) for k in base_weights} # Equal if all zero or NaN
    
    return {factor: (weight / total_weight) for factor, weight in adj_weights.items()}

# --- Main V2 Recommendation Orchestration ---

def recommend_schools_v2(student_profile: Dict, program_query: str, preferences: Dict, number_of_recommendations: int) -> Tuple[pd.DataFrame, int]:
    print(f"V2 Recommendations called. Program: {program_query}, Student GPA: {student_profile.get('gpa')}, Academic Focus: {preferences.get('academic_focus')}")
    
    if DF_PROGRAMS_V2.empty or DF_COLLEGES_V2.empty:
        print("Error: V2 DataFrames (DF_PROGRAMS_V2 or DF_COLLEGES_V2) are empty. Ensure initialize_v2_data() is called correctly at app startup.")
        return pd.DataFrame(), 0
    
    print(f"Initial DF_PROGRAMS_V2 columns: {DF_PROGRAMS_V2.columns.tolist()}") # DEBUG
    print(f"Initial DF_COLLEGES_V2 columns: {DF_COLLEGES_V2.columns.tolist()}") # DEBUG
    if 'school_name_clean' not in DF_PROGRAMS_V2.columns:
        print("CRITICAL ERROR: 'school_name_clean' not in DF_PROGRAMS_V2 at the start of recommend_schools_v2.")
        # Attempt to create it if 'name' or 'standard_college' exists, mirroring V1 loader
        if 'standard_college' in DF_PROGRAMS_V2.columns:
            DF_PROGRAMS_V2['school_name_clean'] = DF_PROGRAMS_V2['standard_college'].astype(str).str.lower().str.strip()
            print("Attempted to create 'school_name_clean' in DF_PROGRAMS_V2 from 'standard_college'.")
        elif 'name' in DF_PROGRAMS_V2.columns:
            DF_PROGRAMS_V2['school_name_clean'] = DF_PROGRAMS_V2['name'].astype(str).str.lower().str.strip()
            print("Attempted to create 'school_name_clean' in DF_PROGRAMS_V2 from 'name'.")
        else:
            return pd.DataFrame(),0 # Still cannot proceed
            
    if 'name_clean' not in DF_COLLEGES_V2.columns:
        print("CRITICAL ERROR: 'name_clean' not in DF_COLLEGES_V2 at the start of recommend_schools_v2.")
        if 'name' in DF_COLLEGES_V2.columns:
            DF_COLLEGES_V2['name_clean'] = DF_COLLEGES_V2['name'].astype(str).str.lower().str.strip()
            print("Attempted to create 'name_clean' in DF_COLLEGES_V2 from 'name'.")
        else:
            return pd.DataFrame(),0 # Still cannot proceed

    # --- Phase 1: Candidate Pool Generation ---
    matched_program_name: Optional[str] = None
    candidate_school_program_pairs: List[Tuple[str, str]] = []

    program_query_clean = program_query.lower().strip()
    exact_matches_df = DF_PROGRAMS_V2[DF_PROGRAMS_V2["program"].astype(str).str.lower().str.strip() == program_query_clean]

    if not exact_matches_df.empty:
        matched_program_name = exact_matches_df["program"].iloc[0]
        for _, row in exact_matches_df.iterrows():
            candidate_school_program_pairs.append((str(row["school_name_clean"]), matched_program_name))
        print(f"Exact program match: {matched_program_name}. Found {len(candidate_school_program_pairs)} initial school-program pairs.")
    elif THEFUZZ_AVAILABLE and preferences.get("allow_fuzzy_program_match", False):
        program_choices = DF_PROGRAMS_V2["program"].dropna().unique()
        if len(program_choices) > 0:
            best_match_tuple = fuzz_process.extractOne(program_query, program_choices, score_cutoff=85)
            if best_match_tuple:
                matched_program_name = best_match_tuple[0]
                fuzzy_matches_df = DF_PROGRAMS_V2[DF_PROGRAMS_V2["program"] == matched_program_name]
                for _, row in fuzzy_matches_df.iterrows():
                    candidate_school_program_pairs.append((str(row["school_name_clean"]), matched_program_name))
                print(f"Fuzzy program match: {matched_program_name} (Score: {best_match_tuple[1]}). Candidates: {len(candidate_school_program_pairs)}")
            else:
                print(f"No suitable fuzzy program match for: {program_query}")
                return pd.DataFrame(), 0
        else:
            print("No programs available for fuzzy matching.")
            return pd.DataFrame(), 0
    else:
        print(f"No exact program match for: {program_query}. Fuzzy matching not enabled/available or no choices.")
        return pd.DataFrame(), 0

    if not candidate_school_program_pairs:
        print("No candidate school-program pairs found.")
        return pd.DataFrame(), 0

    candidate_df = pd.DataFrame(list(set(candidate_school_program_pairs)), columns=["school_name_clean", "program_name"])
    
    # Ensure DF_COLLEGES_V2 has 'name_clean' for the upcoming merge
    if 'name_clean' not in DF_COLLEGES_V2.columns and 'name' in DF_COLLEGES_V2.columns:
        DF_COLLEGES_V2['name_clean'] = DF_COLLEGES_V2['name'].astype(str).str.lower().str.strip()
        print("Dynamically created 'name_clean' in DF_COLLEGES_V2 for merge.")
    elif 'name_clean' not in DF_COLLEGES_V2.columns:
        print("CRITICAL ERROR in V2: DF_COLLEGES_V2 missing 'name_clean' for merge and cannot create it.")
        return pd.DataFrame(), 0
        
    # Add program-specific earnings to candidate_df first
    if 'school_name_clean' not in DF_PROGRAMS_V2.columns:
        # This was checked at the beginning, but double-check before use if logic is complex
        print("CRITICAL ERROR in V2: DF_PROGRAMS_V2 missing 'school_name_clean' column for earnings merge.")
        return pd.DataFrame(), 0
        
    program_earnings_df = DF_PROGRAMS_V2[DF_PROGRAMS_V2["program"] == matched_program_name][
        ["school_name_clean", "earn_mdn_1yr", "earn_mdn_5yr"]
    ].drop_duplicates(subset=["school_name_clean"])

    # Merge candidates with their program earnings (merged_df will have school_name_clean from candidate_df)
    merged_df = pd.merge(candidate_df, program_earnings_df, on="school_name_clean", how="left")
    
    # Then merge with all college data
    # DF_COLLEGES_V2 uses 'name_clean', merged_df (from candidate_df) uses 'school_name_clean'
    merged_df = pd.merge(merged_df, DF_COLLEGES_V2, left_on="school_name_clean", right_on="name_clean", how="inner", suffixes=("_prog_earn", "_coll"))
    
    # After this merge, if 'school_name_clean' was the left key and 'name_clean' the right,
    # pandas might keep both or rename one if the other also existed from the left table.
    # We need to ensure 'school_name_clean' (the one used consistently for school identification) is present.
    # If 'school_name_clean' from the left side (candidate_df) is gone and we have 'name_clean' from right (colleges_df), we might need to use name_clean.
    # However, the critical key used for further processing in score calculations is from the *original* candidate_df, which is `school_name_clean`.

    print(f"V2 Merged_df columns after all merges: {merged_df.columns.tolist()}") 
    if 'school_name_clean' not in merged_df.columns:
        print("Warning in V2: 'school_name_clean' is NOT in merged_df columns directly after merges.")
        # This might happen if suffixes completely eliminate the original key or if it was only on right and not brought over.
        # Given the left_on/right_on, school_name_clean from left should persist.
        # If DF_COLLEGES_V2 also had a school_name_clean, then suffixes would apply.
        # The left_on key `school_name_clean` should persist. If it does not, it implies an issue with the merge itself or prior dataframes.
        # For now, we rely on school_name_clean from candidate_df being the primary key throughout.
        if 'name_clean' in merged_df.columns and 'school_name_clean' not in merged_df.columns:
             print("'school_name_clean' absent, but 'name_clean' (from colleges) is present. This indicates a potential issue if subsequent logic relies on the original key name strictly.")
             # Forcing it for now, but this implies `candidate_df.school_name_clean` might have been lost or unexpectedly renamed
             # merged_df['school_name_clean'] = merged_df['name_clean'] # This is risky if not careful

    initial_candidate_count = len(merged_df)

    # Academic Viability Pre-filter (Example)
    # student_gpa = student_profile.get('gpa')
    # if student_gpa:
    #    merged_df = merged_df[merged_df['average_gpa'].isnull() | (merged_df['average_gpa'] >= student_gpa - 1.0)] 

    if merged_df.empty:
        print("No candidates remaining after merging with college data or pre-filtering.")
        return pd.DataFrame(), 0
    print(f"Total candidates after merging: {len(merged_df)}")

    # --- Phase 2: Multi-Factor Scoring & Ranking ---
    # Pre-calculate normalization stats from the current candidate pool
    norm_stats_earn_1yr = {"min": merged_df["earn_mdn_1yr"].min(), "max": merged_df["earn_mdn_1yr"].max()}
    norm_stats_earn_5yr = {"min": merged_df["earn_mdn_5yr"].min(), "max": merged_df["earn_mdn_5yr"].max()}
    norm_stats_net_price = {"min": merged_df["average_net_price"].min(), "max": merged_df["average_net_price"].max()}
    
    temp_hirer_counts = [len(SCHOOL_TO_COMPANIES_V2.get(snc, {})) for snc in merged_df["school_name_clean"]]
    norm_stats_hirers = {"min": np.min(temp_hirer_counts) if temp_hirer_counts else 0, "max": np.max(temp_hirer_counts) if temp_hirer_counts else 0}

    base_weights = {
        "academic": 0.25, "program_outcome": 0.30, "affordability": 0.15,
        "location": 0.10, "selectivity": 0.05, "environment": 0.05, "career_ops": 0.10
    }
    adjusted_weights = determine_composite_weights(base_weights, preferences)
    
    results = []
    for _, row in merged_df.iterrows():
        school_data = row.to_dict()
        school_name_clean = str(school_data["school_name_clean"])
        
        # Data for scoring
        school_avg_gpa = pd.to_numeric(school_data.get("average_gpa"), errors='coerce')
        school_avg_sat = pd.to_numeric(school_data.get("average_sat_composite"), errors='coerce')
        program_earn_1yr = pd.to_numeric(school_data.get("earn_mdn_1yr"), errors='coerce')
        program_earn_5yr = pd.to_numeric(school_data.get("earn_mdn_5yr"), errors='coerce')
        school_net_price = pd.to_numeric(school_data.get("average_net_price"), errors='coerce')
        school_state_full = str(school_data.get("location", ""))
        school_state = school_state_full.split(",")[-1].strip() if pd.notna(school_state_full) and "," in school_state_full else None
        school_acceptance_rate = pd.to_numeric(school_data.get("acceptance_rate"), errors='coerce')
        school_type = str(school_data.get("type_of_institution", ""))
        school_num_students = pd.to_numeric(school_data.get("number_of_students"), errors='coerce')

        # Calculate scores
        s_academic = calculate_academic_fit_score(
            student_profile["gpa"], student_profile.get("effective_sat"),
            school_avg_gpa, school_avg_sat,
            preferences.get("academic_focus", "match")
        )
        s_program_outcome = calculate_program_outcome_score(
            program_earn_1yr, program_earn_5yr,
            norm_stats_earn_1yr, norm_stats_earn_5yr
        )
        s_affordability = calculate_affordability_score(
            school_net_price, preferences.get("cost",{}).get("max_net_price_per_year"),
            norm_stats_net_price
        )
        s_location = calculate_location_score(
            school_state, None, # school_city not used yet
            preferences.get("location", {}).get("states"), preferences.get("location", {}).get("region")
        )
        s_selectivity = calculate_selectivity_score(
            school_acceptance_rate, preferences.get("selectivity_preference", "any")
        )
        s_environment = calculate_environment_score(
            school_type, school_num_students, # Pass num_students directly
            preferences.get("school_type"), preferences.get("school_size")
        )
        s_career = calculate_career_opportunities_score(school_name_clean, norm_stats_hirers)

        # Ensure essential numeric fields from school_data are pre-converted for Pydantic
        # This helps prevent Pydantic validation errors if raw CSV data slips through
        school_data["average_gpa"] = pd.to_numeric(school_data.get("average_gpa"), errors='coerce')
        school_data["average_sat_composite"] = pd.to_numeric(school_data.get("average_sat_composite"), errors='coerce')
        school_data["earn_mdn_1yr"] = pd.to_numeric(school_data.get("earn_mdn_1yr"), errors='coerce')
        school_data["earn_mdn_5yr"] = pd.to_numeric(school_data.get("earn_mdn_5yr"), errors='coerce')
        school_data["average_net_price"] = pd.to_numeric(str(school_data.get("average_net_price")).replace('$','').replace(',',''), errors='coerce')
        
        raw_enrollment = school_data.get("number_of_students")
        if pd.notna(raw_enrollment):
            school_data["number_of_students"] = pd.to_numeric(str(raw_enrollment).replace(',',''), errors='coerce')
        else:
            school_data["number_of_students"] = np.nan

        raw_acceptance_rate = school_data.get("acceptance_rate")
        if pd.notna(raw_acceptance_rate):
            rate_str = str(raw_acceptance_rate).replace('%','')
            numeric_rate = pd.to_numeric(rate_str, errors='coerce')
            if pd.notna(numeric_rate) and numeric_rate > 1: # Assuming it's a percentage like 50 not 0.50
                school_data["acceptance_rate"] = numeric_rate / 100.0
            elif pd.notna(numeric_rate): # Already a decimal
                school_data["acceptance_rate"] = numeric_rate
            else:
                school_data["acceptance_rate"] = np.nan # Coercion failed
        else:
            school_data["acceptance_rate"] = np.nan
            
        # UGDS and demographic percentages should ideally also be cleaned if sourced from raw strings
        school_data["UGDS"] = pd.to_numeric(str(school_data.get("UGDS")).replace(',',''), errors='coerce')
        school_data["UGDS_WHITE"] = pd.to_numeric(school_data.get("UGDS_WHITE"), errors='coerce') # Assuming these are already decimals or NaN
        school_data["UGDS_BLACK"] = pd.to_numeric(school_data.get("UGDS_BLACK"), errors='coerce')
        school_data["UGDS_HISP"] = pd.to_numeric(school_data.get("UGDS_HISP"), errors='coerce')
        school_data["UGDS_ASIAN"] = pd.to_numeric(school_data.get("UGDS_ASIAN"), errors='coerce')

        composite_score = (
            s_academic * adjusted_weights["academic"] +
            s_program_outcome * adjusted_weights["program_outcome"] +
            s_affordability * adjusted_weights["affordability"] +
            s_location * adjusted_weights["location"] +
            s_selectivity * adjusted_weights["selectivity"] +
            s_environment * adjusted_weights["environment"] +
            s_career * adjusted_weights["career_ops"]
        )
        
        result_item = {**school_data} 
        result_item.update({
            "program_name": matched_program_name, # Ensure this is the canonical matched program
            "composite_score": composite_score,
            "s_academic": s_academic, "s_program_outcome": s_program_outcome, "s_affordability": s_affordability,
            "s_location": s_location, "s_selectivity": s_selectivity, "s_environment": s_environment, "s_career": s_career
        })
        results.append(result_item)

    if not results:
        print("No schools after scoring.")
        return pd.DataFrame(), 0

    ranked_df = pd.DataFrame(results).sort_values(by="composite_score", ascending=False)

    # --- Phase 3: Post-Processing & Output ---
    final_recommendations = ranked_df.head(number_of_recommendations).copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Add "Why this school?" snippets
    # More sophisticated logic needed here based on dominant scores vs. preferences
    def generate_why_snippet(row):
        reasons = []
        # Check for existence and ensure scores are not NaN before comparison
        if pd.notna(row.get('s_academic')) and row.get('s_academic', 0) > 0.7: reasons.append("strong academic fit")
        if pd.notna(row.get('s_program_outcome')) and row.get('s_program_outcome', 0) > 0.7: reasons.append("good program outcomes")
        if pd.notna(row.get('s_affordability')) and row.get('s_affordability', 0) > 0.7: reasons.append("affordable option")
        if not reasons: return "Overall good match based on your profile."
        return "Highlights: " + ", ".join(reasons[:2]) + "."
    
    if not final_recommendations.empty:
        final_recommendations["Why_This_School_Snippet"] = final_recommendations.apply(generate_why_snippet, axis=1)
    else:
        final_recommendations["Why_This_School_Snippet"] = pd.Series(dtype='str')

    # Define V2 Tiers (example)
    def assign_v2_tier(score):
        if score >= 0.8: return "Excellent Fit"
        if score >= 0.7: return "Strong Fit"
        if score >= 0.55: return "Good Fit"
        if score >= 0.4: return "Possible Fit"
        return "Consider Alternatives"
    
    if not final_recommendations.empty:
        final_recommendations["V2_Recommendation_Tier"] = final_recommendations["composite_score"].apply(assign_v2_tier)
    else:
        final_recommendations["V2_Recommendation_Tier"] = pd.Series(dtype='str')

    if not final_recommendations.empty:
        final_recommendations["Admission_Statistics"] = final_recommendations["school_name_clean"].apply(
            lambda snc: get_admission_statistics_v2(snc) if pd.notna(snc) else [] 
        )

    print(f"V2: Returning {len(final_recommendations)} recommendations.")
    return final_recommendations, len(merged_df)

# Helper for V2 to get admission stats from its own data dict
def get_admission_statistics_v2(std_name_clean: str) -> List[Dict[str, Any]]:
    if std_name_clean in ADMISSIONS_DATA_DICT_V2:
        stats_list = []
        for year in sorted(ADMISSIONS_DATA_DICT_V2[std_name_clean].keys()):
            metrics = ADMISSIONS_DATA_DICT_V2[std_name_clean][year]
            stats_list.append({
                "year": year,
                "metrics": metrics
            })
        return stats_list
    return []

# Note: This is a skeleton with initial implementations. Many details (normalization specifics, 
# score function shapes, weight adjustments, "Why this school" logic, exact output formatting for RecommendationV2)
# may need further refinement and robust error handling during full development. 