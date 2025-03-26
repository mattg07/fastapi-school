import pandas as pd
import numpy as np

############################################################
# Load Data
############################################################
def load_data():
    # Load CSV files. Adjust paths as needed.
    school_programs = pd.read_csv("programs_cleaned.csv")
    alumni_data = pd.read_csv("companies_data_cleaned.csv")
    university_data = pd.read_csv("universities_cleaned.csv")

    # Convert numeric row index to a column for reference
    university_data = university_data.reset_index().rename(columns={"index": "university_id"})

    return school_programs, alumni_data, university_data

############################################################
# Get Available Programs
############################################################
def get_available_programs(school_programs):
    return school_programs["program"].unique()

############################################################
# Merge Data
############################################################
def merge_data(school_programs, university_data):
    # Standardize names for merging
    university_data["standard_name"] = university_data["standard_name"].str.lower()
    school_programs["standard_college"] = school_programs["standard_college"].str.lower()

    # Merge on standard name
    merged_data = pd.merge(
        school_programs,
        university_data,
        left_on="standard_college",
        right_on="standard_name",
        how="inner"
    )

    # Convert 'PS' or other strings in earn_mdn_1yr/earn_mdn_5yr to NaN
    merged_data["earn_mdn_1yr"] = pd.to_numeric(merged_data["earn_mdn_1yr"], errors="coerce")
    merged_data["earn_mdn_5yr"] = pd.to_numeric(merged_data["earn_mdn_5yr"], errors="coerce")

    return merged_data

############################################################
# Ranking by Academics
############################################################
def rank_by_academics(merged_data):
    # Compute academic_score
    merged_data["academic_score"] = (
        (merged_data["niche_communications_sat_low"] + merged_data["niche_communications_sat_high"]) / 2
    ) + merged_data["us_news_detail_high_school_gpa"]

    # Sort by academic_score desc
    ranked = merged_data.sort_values("academic_score", ascending=False, na_position="last").copy()
    # Rename
    ranked.rename(columns={"academic_score": "academic_score_acad"}, inplace=True)

    return ranked

############################################################
# Ranking by Salary
############################################################
def rank_by_salary(merged_data):
    # Sort by 5yr salary desc
    ranked_salary = merged_data.sort_values("earn_mdn_5yr", ascending=False, na_position="last").copy()
    # Rename columns
    ranked_salary.rename(
        columns={"earn_mdn_1yr": "earn_mdn_1yr_sal", "earn_mdn_5yr": "earn_mdn_5yr_sal"},
        inplace=True
    )
    return ranked_salary

############################################################
# Alumni Presence
############################################################
def alumni_presence(alumni_data, relevant_schools):
    # We'll track all relevant schools from relevant_schools
    school_set = set(relevant_schools["standard_college"].unique())

    alumni_list = []
    for _, row in alumni_data.iterrows():
        company_name = row["company"]
        # up to 15 pairs: school_i, alumnicount_i
        for i in range(1, 16):
            s_col = f"school_{i}"
            c_col = f"alumnicount_{i}"

            school_lower = str(row.get(s_col, "")).lower().strip()
            count_val = row.get(c_col, 0)

            if school_lower in school_set and school_lower:
                alumni_list.append({
                    "company": company_name,
                    "standard_college": school_lower,
                    "alumni_count": count_val,
                })

    # Convert to DF
    df_alumni = pd.DataFrame(alumni_list)

    if df_alumni.empty:
        return df_alumni

    # Aggregate all alumni counts for each school
    df_agg = df_alumni.groupby("standard_college", as_index=False).agg({
        "alumni_count": "sum"
    })

    return df_agg

############################################################
# Combine final table
############################################################
def compile_final_table(academic_rank, salary_rank, alumni_df):
    # Merge academic + salary
    final_table = academic_rank.merge(
        salary_rank,
        on=["standard_name", "program", "standard_college"],
        how="inner"
    )

    # Merge with aggregated alumni on standard_college
    final_table = final_table.merge(
        alumni_df,
        on="standard_college",
        how="left"
    )

    # Now, deduplicate if needed by standard_name
    final_table = final_table.sort_values("earn_mdn_5yr_sal", ascending=False)
    final_table = final_table.drop_duplicates(subset=["standard_name"], keep="first")

    return final_table

############################################################
# Personalized Ranking
#   Weighted difference approach.
############################################################
def personalized_ranking(final_table, student_gpa, student_sat):
    # 1) We'll isolate the average SAT
    if (
        "niche_communications_sat_low" in final_table.columns and
        "niche_communications_sat_high" in final_table.columns and
        "us_news_detail_high_school_gpa" in final_table.columns
    ):
        final_table["avg_sat"] = (
            final_table["niche_communications_sat_low"] + final_table["niche_communications_sat_high"]
        ) / 2.0
        final_table["avg_gpa"] = final_table["us_news_detail_high_school_gpa"]

        # 2) Filter out schools far from student's SAT/GPA (like +/- 150 SAT, +/- 0.3 GPA)
        sat_threshold = 150
        gpa_threshold = 0.3
        filtered = final_table[
            (final_table["avg_sat"].notnull()) &
            (abs(final_table["avg_sat"] - student_sat) <= sat_threshold) &
            (final_table["avg_gpa"].notnull()) &
            (abs(final_table["avg_gpa"] - student_gpa) <= gpa_threshold)
        ].copy()
    else:
        # If columns missing, fallback to entire final_table
        filtered = final_table.copy()
        filtered["avg_sat"] = np.nan
        filtered["avg_gpa"] = np.nan

    # 3) Weighted difference approach
    #    sat_diff = |avg_sat - student_sat| / 400
    #    gpa_diff = |avg_gpa - student_gpa| / 1.0
    #    total_diff = w_sat * sat_diff + w_gpa * gpa_diff
    #    match_score = 1 - total_diff  (larger is better)

    w_sat = 0.6
    w_gpa = 0.4

    # For any rows missing avg_sat or avg_gpa, we'll fill them with the student's.
    # That yields zero difference, but they'll be included. Alternatively, we can drop.
    filtered["sat_diff"] = abs(filtered["avg_sat"].fillna(student_sat) - student_sat) / 400.0
    filtered["gpa_diff"] = abs(filtered["avg_gpa"].fillna(student_gpa) - student_gpa) / 1.0

    filtered["total_diff"] = w_sat * filtered["sat_diff"] + w_gpa * filtered["gpa_diff"]
    filtered["student_match_score"] = 1.0 - filtered["total_diff"]

    # 4) Sort by descending match_score
    filtered = filtered.sort_values("student_match_score", ascending=False)
    return filtered

############################################################
# Main Recommendation Logic
############################################################
def recommend_schools(program, merged_data, alumni_data, student_gpa, student_sat):
    # Filter merged data for relevant program
    relevant_schools = merged_data[
        merged_data["program"].str.lower() == program.lower()
    ]

    if relevant_schools.empty:
        print("No matching programs found.")
        return None, None, None, None, None

    # 1) Rank academically
    academic_rank = rank_by_academics(relevant_schools)

    # 2) Rank by salary
    salary_rank = rank_by_salary(relevant_schools)

    # 3) Build aggregated alumni df
    alumni_df = alumni_presence(alumni_data, relevant_schools)

    # 4) Merge all, dropping duplicates
    final_table = compile_final_table(academic_rank, salary_rank, alumni_df)

    # 5) Weighted difference approach for personalized ranking
    personalized = personalized_ranking(final_table, student_gpa, student_sat)

    return academic_rank, salary_rank, alumni_df, final_table, personalized

############################################################
# CLI
############################################################
if __name__ == "__main__":
    school_programs, alumni_data, university_data = load_data()
    merged_data = merge_data(school_programs, university_data)

    available_programs = get_available_programs(school_programs)
    print("Available Programs:")
    print("\n".join(available_programs))

    program = input("Enter your desired field of study: ")
    student_gpa = float(input("Enter your GPA: "))
    student_sat = int(input("Enter your SAT Score: "))

    if program not in available_programs:
        print("Invalid program choice. Please select from the provided list.")
    else:
        academic_rank, salary_rank, alumni_df, final_table, student_recommendation = recommend_schools(
            program, merged_data, alumni_data, student_gpa, student_sat
        )

        if academic_rank is not None:
            print("\nRanking by Academics (Top 10):")
            print(
                academic_rank[[
                    "standard_name",
                    "niche_communications_sat_low",
                    "niche_communications_sat_high",
                    "us_news_detail_high_school_gpa",
                    "academic_score_acad",
                ]].head(10).reset_index(drop=True)
            )

            print("\nRanking by Salary (Top 10):")
            print(
                salary_rank[[
                    "standard_name",
                    "earn_mdn_1yr_sal",
                    "earn_mdn_5yr_sal",
                ]].head(10).reset_index(drop=True)
            )

            print("\nAggregated Alumni Info:")
            print(alumni_df.head(10).reset_index(drop=True))

            print("\nFinal Combined Ranking Table (deduplicated) (Top 10):")
            print(
                final_table[[
                    "standard_name",
                    "program",
                    "academic_score_acad",
                    "earn_mdn_5yr_sal",
                    "alumni_count",
                ]].head(10).reset_index(drop=True)
            )

            print("\nPersonalized Recommendations (Top 10):")
            print(
                student_recommendation[[
                    "standard_name",
                    "program",
                    "academic_score_acad",
                    "earn_mdn_5yr_sal",
                    "student_match_score",
                ]].head(10).reset_index(drop=True)
            )