import pandas as pd
import numpy as np
import argparse

def numeric_or_nan(x):
    """
    Attempt to convert to float. If it fails, return NaN.
    """
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
    path_companies: str = "companies_data_cleaned.csv"
):
    """
    Recommends schools based on user-provided GPA, SAT, and program of interest.
    """
    # -----------------------------
    # 1. LOAD THE DATA
    # -----------------------------
    df_colleges = pd.read_csv(path_colleges)
    df_programs = pd.read_csv(path_programs)
    df_school_sup = pd.read_csv(path_school_sup)
    df_companies = pd.read_csv(path_companies)

    # Coerce SAT columns to numeric, converting invalid strings to NaN
    df_colleges["average_sat_composite"] = pd.to_numeric(df_colleges["average_sat_composite"], errors="coerce")
    df_school_sup["SAT_AVG"] = pd.to_numeric(df_school_sup["SAT_AVG"], errors="coerce")

    # -----------------------------
    # 2. NORMALIZE MERGE KEYS
    # -----------------------------
    df_colleges["name_clean"] = df_colleges["name"].str.lower().str.strip()

    # In df_programs, we rely on either 'standard_college' or 'name':
    if "standard_college" in df_programs.columns:
        df_programs["school_name_clean"] = df_programs["standard_college"].str.lower().str.strip()
    else:
        df_programs["school_name_clean"] = df_programs["name"].str.lower().str.strip()

    # For the supplemental data, unify to "school_name_clean"
    df_school_sup["school_name_clean"] = df_school_sup["INSTNM"].str.lower().str.strip()

    # -----------------------------
    # 3. BUILD COMPANY HIRING LOOKUP
    # -----------------------------
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

    # -----------------------------
    # 4. FILTER df_programs BY USER PROGRAM
    # -----------------------------
    user_program_clean = user_program.lower().strip()
    if "program_clean" in df_programs.columns:
        df_prog_filtered = df_programs[df_programs["program_clean"].str.lower() == user_program_clean]
    else:
        df_prog_filtered = df_programs[df_programs["program"].str.lower() == user_program_clean]

    if df_prog_filtered.empty:
        print(f"No schools found offering program '{user_program}'.")
        return pd.DataFrame()

    # -----------------------------
    # 5. MERGE PROGRAM DATA WITH COLLEGES & SUP
    # -----------------------------
    merged = pd.merge(
        df_prog_filtered,
        df_colleges,
        left_on="school_name_clean",
        right_on="name_clean",
        how="inner",
        suffixes=("_prog", "_col")
    )

    if "name_col" in merged.columns:
        merged.rename(columns={"name_col": "name"}, inplace=True)

    merged = pd.merge(
        merged,
        df_school_sup,
        on="school_name_clean",
        how="inner"
    )

    # -----------------------------
    # 6. APPLY GPA & SAT FILTERS
    # -----------------------------
    def within_gpa_range(row):
        if pd.isnull(row["average_gpa"]):
            return False
        return abs(row["average_gpa"] - user_gpa) <= 0.3

    def within_sat_range(row):
        sat_val = row["average_sat_composite"]
        if pd.isnull(sat_val):
            sat_val = row["SAT_AVG"]
        if pd.isnull(sat_val):
            return False
        return abs(sat_val - user_sat) <= 150

    merged["gpa_in_range"] = merged.apply(within_gpa_range, axis=1)
    merged["sat_in_range"] = merged.apply(within_sat_range, axis=1)

    final_candidates = merged[
        (merged["gpa_in_range"] == True) &
        (merged["sat_in_range"] == True)
    ].copy()

    if final_candidates.empty:
        print("No schools are within the specified GPA/SAT range for that program.")
        return pd.DataFrame()

    # -----------------------------
    # 7. PARSE ADMISSION RATE & NET PRICE
    # -----------------------------
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

    # -----------------------------
    # 8. CONVERT EARNINGS COLUMNS & SORT
    # -----------------------------
    final_candidates.loc[:, "earn_mdn_1yr_num"] = final_candidates["earn_mdn_1yr"].apply(numeric_or_nan)
    final_candidates.loc[:, "earn_mdn_5yr_num"] = final_candidates["earn_mdn_5yr"].apply(numeric_or_nan)

    schools_with_salary = final_candidates[final_candidates["earn_mdn_1yr_num"].notna()].copy()
    schools_without_salary = final_candidates[final_candidates["earn_mdn_1yr_num"].isna()].copy()

    schools_with_salary = schools_with_salary.sort_values(
        by=["earn_mdn_1yr_num"],
        ascending=False
    )
    schools_without_salary = schools_without_salary.sort_values(
        by=["acceptance_rate_numeric"],
        ascending=True,
        na_position="last"
    )

    final_candidates = pd.concat([schools_with_salary, schools_without_salary])

    # -----------------------------
    # 9. LOOKUP COMPANIES FOR EACH SCHOOL
    # -----------------------------
    def get_companies_for_school(row):
        school_name_clean = row["name_clean"]
        if school_name_clean in school_to_companies:
            return list(school_to_companies[school_name_clean])
        else:
            return []

    final_candidates["fortune_500_companies"] = final_candidates.apply(get_companies_for_school, axis=1)

    # -----------------------------
    # 10. BUILD OUTPUT
    # -----------------------------
    if "name" not in final_candidates.columns:
        print("Error: Could not find 'name' column after merges.")
        return pd.DataFrame()

    recommended_cols = [
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
        "acceptance_rate_numeric"
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
            "average_net_price_numeric": "Avg_Net_Price"
        },
        inplace=True
    )

    recommendations["Has_Salary_Data"] = recommendations["Median_Earnings_1yr"].notna()

    column_order = [
        "School",
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
        "Longitude"
    ]
    recommendations = recommendations[column_order]

    recommendations.reset_index(drop=True, inplace=True)
    return recommendations

# -----------------------------
#  Command Line Interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recommend schools based on GPA, SAT score, and program of interest."
    )
    parser.add_argument(
        "--gpa",
        type=float,
        required=True,
        help="Your GPA (e.g., 3.7)"
    )
    parser.add_argument(
        "--sat",
        type=int,
        required=True,
        help="Your SAT score (e.g., 1400)"
    )
    parser.add_argument(
        "--program",
        type=str,
        required=True,
        help="Field of study/program (e.g., 'Computer Science')"
    )
    
    args = parser.parse_args()

    df_result = recommend_schools(
        user_gpa=args.gpa,
        user_sat=args.sat,
        user_program=args.program
    )

    if not df_result.empty:
        output_file = "ranked_by_earnings.csv"
        df_result.to_csv(output_file, index=False)
        
        total_schools = len(df_result)
        schools_with_salary = df_result["Has_Salary_Data"].sum()
        
        print(f"\nResults Summary:")
        print(f"Total schools matching criteria: {total_schools}")
        print(f"Schools with salary data: {schools_with_salary}")
        print(f"Schools without salary data: {total_schools - schools_with_salary}")
        print(f"\nDetailed results have been written to: {output_file}")
    else:
        print("No recommendations found.")
