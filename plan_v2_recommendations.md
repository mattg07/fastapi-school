# Plan: V2 University Recommendation System

## 1. Introduction & Goals

This document outlines the plan for creating a Version 2 (V2) University Recommendation System. The primary goal is to deliver a more robust, nuanced, and valuable advisory tool for students by moving beyond the current primarily rule-based academic fit model to a multi-factor scoring and ranking system.

**Key Objectives:**
-   Provide more personalized and context-aware recommendations.
-   Incorporate a wider range of factors beyond just academic scores and overall school-level earnings.
-   Offer clearer explanations for why certain schools are recommended.
-   Improve the underlying data handling and matching logic.
-   Address limitations of the V1 system, such as rigidity and reliance on broad averages.
-   Establish a foundation for a production-quality recommendation product.

## 2. Proposed Technical Approach: Hybrid Model

The V2 system will employ a hybrid approach:

1.  **Phase 1: Candidate Pool Generation:**
    *   **Program Matching:** Expect validated program names from the API request (assuming frontend uses a dropdown of official Dept. of Ed. categories). The service will perform an exact, case-insensitive match. (See Section 4.2 for contingency if free-text input is used).
    *   **Academic Viability Pre-filtering:** Apply initial broad academic filters. This could involve minimum thresholds (e.g., if a student's GPA is far below any reasonable chance of admission even for a "Far Reach" school) or allow students to specify a "minimum academic bar" they are willing to consider. This phase aims to reduce the set of schools to a manageable number for more intensive scoring.
2.  **Phase 2: Multi-Factor Scoring & Ranking:**
    *   For each unique school-program combination in the candidate pool, calculate a **Composite Suitability Score**.
    *   This score will be a weighted sum of several normalized sub-scores representing different dimensions of "fit" and "value."
3.  **Phase 3: Personalization, Post-Processing & Output:**
    *   User preferences (e.g., for cost, location, school size, career outcomes) will dynamically adjust the weights of the sub-scores.
    *   Rank schools based on the final Composite Suitability Score.
    *   Provide clear tier labels (e.g., "Strong Fit," "Good Fit," "Ambitious Fit," "Possible Fit") based on score ranges and dominant factors.
    *   Include "Why this school?" snippets highlighting key strengths relevant to the student's profile and preferences.
    *   **Frontend Compatibility:** Ensure that V2 API responses maintain the same field names and data structures for data points consumed by the existing V1 frontend, while new fields can be added.

## 3. Data Inventory and Utilization Strategy

The system will continue to leverage the existing CSV data sources, loaded into pandas DataFrames at startup. Their utilization will be more targeted for the new scoring components.

| Data Source                          | Relevant Columns (Examples)                                                                                                                              | V2 Utilization Strategy                                                                                                                                                                                                                                                                                                                        |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `colleges_data_cleaned.csv`        | `name`, `name_clean`, `location`, `average_gpa`, `average_sat_composite`, `average_act_composite`, `acceptance_rate`, `type_of_institution`, `level_of_institution`, `average_net_price`, `number_of_students`, `LATITUDE`, `LONGITUDE` | - **Academic Fit Score:** `average_gpa`, `average_sat_composite`, `average_act_composite`. <br> - **Affordability Score:** `average_net_price`. <br> - **Location Score:** `location` (for State matching), `LATITUDE`, `LONGITUDE`. `Region` matching requires a new internal state-to-region map. `Setting` (urban/rural) preference is removed due to data constraints. <br> - **Selectivity Score:** `acceptance_rate`. <br> - **School Environment Score:** `type_of_institution`, `level_of_institution`, `number_of_students` (thresholds needed for small/medium/large). <br> - Base for school display name and joining. |
| `programs_cleaned.csv`             | `school_name_clean`, `program`, `earn_mdn_1yr`, `earn_mdn_5yr`                                                                                             | - **Program Matching:** `program` (exact match assumed from validated input). <br> - **Program Strength/Outcome Score:** Program-specific `earn_mdn_1yr`, `earn_mdn_5yr`. This is critical for linking recommendations to specific program outcomes.                                                                                                 |
| `school_sup_data_cleaned.csv`      | `INSTNM`, `school_name_clean`, `UGDS`, `UGDS_WHITE` (and other demographics), `SAT_AVG`, `ADM_RATE`                                                            | - Supplements enrollment data (`UGDS`). <br> - Provides demographic data (can be part of School Environment Score or additional info). <br> - Fallback for `SAT_AVG` and `ADM_RATE` if missing in `DF_COLLEGES`.                                                                                                                            |
| `companies_data_cleaned.csv`       | (Processed to `SCHOOL_TO_COMPANIES` map: `{school_name_clean: {company: count}}`)                                                                          | - **Career Opportunities Score:** Quantifies hiring strength from (Fortune 500) companies for a given school. Could be enhanced if program-specific hirer data becomes available.                                                                                                                                                 |
| `admission_trends_cleaned.csv`   | (Processed to `admissions_data_dict`)                                                                                                                     | - Provides historical context for admission selectivity. Can be used in "Why this school?" or as a factor in a more advanced Selectivity Score. Included in detailed API output for user insight.                                                                                                                                        |
| `school_plus_image_data_cleaned.csv` | `instnm`, `school_name_clean`, `cdn_url`, `thumb_link`                                                                                                     | - Provides image URLs for front-end display. `school_name_clean` for joining.                                                                                                                                                                                                                                                    |

**Data Normalization:** Numerical data used in scoring (e.g., GPA, SAT, net price, earnings) will be normalized (e.g., min-max scaling or z-score normalization across the relevant dataset) to ensure fair contribution to the composite score before applying weights.

## 4. Core Recommendation Algorithm Details (`/v2/recommendations`)

### 4.1. Input Parameters

The `/v2/recommendations` POST endpoint will accept a JSON payload similar to V1 but with potentially more structured preference inputs:

```json
{
  "student_profile": {
    "gpa": 3.7,
    "sat": 1350, // Or ACT, with conversion logic
    "act": 29   // Optional if SAT provided
  },
  "program_query": "Software Engineering", // User's desired program of study - expected to be an exact name from a predefined list
  "preferences": {
    "academic_focus": "match", // e.g., "match", "challenge_me" (reach-focused), "less_stress" (safety-focused)
    "location": { 
      "states": ["CA", "NY"], // Preferred states
      "region": "Northeast", // Preferred region (requires internal state-to-region mapping)
      // "setting": "urban", // urban, suburban, rural (REMOVED due to data constraints)
      "max_distance_miles": null // (Future: if current location provided)
    },
    "cost": {
      "max_net_price_per_year": 40000, // Max willing to consider
      "importance": "high" // low, medium, high (influences weight)
    },
    "school_size": ["medium", "large"], // small, medium, large (requires defined thresholds for number_of_students)
    "school_type": "public", // public, private, any
    "career_outcomes_importance": "high", // Influences weight of salary/hirer scores
    "selectivity_preference": "moderate" // "any", "moderate", "high_challenge" (influences how admission rate is scored)
  },
  "number_of_recommendations": 10
}
```

### 4.2. Phase 1: Candidate Pool Generation

1.  **Program Matching:**
    *   Take `program_query` from input. This is assumed to be a validated program name from the official list (e.g., Dept. of Education categories, provided to the frontend via `/v2/programs`).
    *   Perform an exact, case-insensitive, stripped match against the `program` names in `DF_PROGRAMS`.
    *   Filter `DF_PROGRAMS` for schools offering the exactly matched program. This yields an initial list of (school\_name\_clean, program\_name) pairs.
    *   **Contingency for Free-Text Program Input:** If the frontend design ultimately allows free-text input for programs (not the primary assumption), this step would need to incorporate fuzzy matching (e.g., using `thefuzz` library with a high similarity threshold like >0.85, after attempting an exact match). This would replace V1's naive length-based `find_program` fallback and require careful tuning to minimize incorrect matches.

2.  **Academic Viability Pre-filter (Optional but Recommended):**
    *   For each school in the program-matched list, retrieve its average GPA/SAT from `DF_COLLEGES`.
    *   Apply a broad filter:
        *   If student's GPA/SAT is drastically below the school's minimum typical range (e.g., more than 1.0 GPA point below or 300 SAT points below – thresholds to be determined and potentially configurable), exclude the school unless the student's `academic_focus` preference indicates a desire for extreme reach.
        *   This step aims to quickly remove clearly non-viable options before more intensive scoring.

### 4.3. Phase 2: Multi-Factor Scoring & Ranking

For each (school\_name\_clean, program\_name) pair in the filtered candidate pool:

**A. Calculate Sub-Scores (Normalized, e.g., 0 to 1):**

1.  **Academic Fit Score (`S_academic`):**
    *   Calculate `gpa_diff = school_avg_gpa - student_gpa`.
    *   Calculate `sat_diff = school_avg_sat - student_effective_sat`.
    *   Convert these differences into a score (0-1). Smaller absolute `diff` generally means better fit (score closer to 1 for "match" focus).
    *   The `academic_focus` preference ("match", "challenge_me", "less_stress") will define the ideal `diff` and the shape of the scoring function (e.g., for "challenge_me", a moderately positive `diff` where school is harder gets a higher score). This requires defining specific scoring functions/curves for each focus.

2.  **Program Strength/Outcome Score (`S_program_outcome`):**
    *   Use program-specific `earn_mdn_1yr` and `earn_mdn_5yr` from `DF_PROGRAMS`.
    *   Normalize these earnings (e.g., min-max scaling across the candidate set or a global baseline from all programs).
    *   Example: `(normalized_earn_1yr * W_1yr_earn) + (normalized_earn_5yr * W_5yr_earn)`.

3.  **Affordability Score (`S_affordability`):**
    *   Based on `average_net_price` from `DF_COLLEGES`.
    *   If student specifies `max_net_price_per_year`, score higher if `average_net_price <= max_net_price_per_year`, decreasing as it exceeds.
    *   If no max price, score based on a general scale (lower price = higher score, normalized).

4.  **Location Score (`S_location`):**
    *   If `states` preference: Score = 1 if school in preferred state, 0 otherwise.
    *   If `region` preference: Score based on match against the internal state-to-region map.
    *   `setting` (urban/rural) preference is **removed** due to data constraints.
    *   Combine into a single location score, possibly weighting state match higher if both state and region are specified.

5.  **School Selectivity Score (`S_selectivity`):**
    *   Based on `acceptance_rate` from `DF_COLLEGES`.
    *   Score mapping depends on `selectivity_preference` ("any", "moderate", "high_challenge"), e.g., for "high_challenge", lower acceptance rates (more selective) get higher scores after normalization.

6.  **School Environment Score (`S_environment`):**
    *   `type_of_institution` (public/private), `level_of_institution` (4-year), `number_of_students` (small/medium/large based on defined thresholds).
    *   Score based on matches with student's `school_size` and `school_type` preferences.

7.  **Career Opportunities Score (`S_career`):**
    *   Use the `SCHOOL_TO_COMPANIES` map.
    *   Score based on the count/diversity of unique Fortune 500 hirers, or a sum of alumni counts, normalized.

**B. Determine Weights (`W_factor`):**

*   Each sub-score factor gets a **default base weight** (e.g., Academic=0.3, ProgramOutcome=0.25, Affordability=0.15, Location=0.1, Selectivity=0.05, Environment=0.05, Career=0.1. These must sum to 1).
*   Student preferences (e.g., `cost.importance = "high"`, `career_outcomes_importance = "high"`) dynamically adjust these base weights (e.g., "high" importance might multiply base weight by 1.5, "low" by 0.5).
*   After adjustment, weights must be **re-normalized** to sum to 1 to maintain consistent score scaling.

**C. Calculate Composite Suitability Score (`Score_composite`):**

`Score_composite = (S_academic * W_academic_adj) + (S_program_outcome * W_program_outcome_adj) + ...`

### 4.4. Phase 3: Post-processing & Output

1.  **Ranking:** Sort all candidate (school, program) pairs by `Score_composite` in descending order.
2.  **Tier Labeling & "Why this School?":**
    *   Define new, descriptive tier labels based on `Score_composite` ranges (e.g., Top Tier: >0.85, Strong Fit: 0.7-0.85) AND/OR dominant `S_factor` values.
        *   Examples: "Excellent Overall Fit (High Academic & Career Scores)", "Strong Program Outcomes, Affordable", "Good Academic Fit, Targets Location".
    *   Generate brief explanations ("Why this school?" snippets) by identifying the top 2-3 sub-scores or factors that contributed most to the composite score for that student.
3.  **Filtering for Diversity (Optional but Recommended for Production):**
    *   If top N results are too homogenous, apply logic to introduce diversity (e.g., in cost, location) from slightly lower-ranked schools.
4.  **Final Output Structure (Frontend Compatibility Focus):**
    *   Return the top `number_of_recommendations`.
    *   **Each recommendation object in the list MUST strive to maintain existing V1 field names and data structures for data points the frontend currently consumes.** This is crucial for backward compatibility.
    *   **New V2 fields (e.g., `Program_Name`, `Composite_Score`, `V2_Tier_Label`, `Why_This_School_Snippet`, individual sub-scores if desired) can be ADDED.**
    *   Example of maintaining V1 fields while adding V2:
        ```json
        {
          "School": "University Name (from V1 logic)", // Display name
          "Program_Name": "Matched Program Name (V2 specific)",
          "Recommendation_Tier": "V1 Tier if still calculable/useful OR new V2 Tier Label", // Decision needed: Keep old field name with new values, or add new V2_Tier_Label
          "V2_Tier_Label": "Excellent Overall Fit (High Academic & Career Scores)", // Example new V2 field
          "Composite_Score": 0.88, // Example new V2 field
          "Why_This_School_Snippet": "Strong in Computer Science earnings and matches your academic profile well.", // Example
          "Has_Salary_Data": true, // V1 field
          "Median_Earnings_1yr": 95000.0, // V1 field (now program-specific)
          "Median_Earnings_5yr": 145000.0, // V1 field (now program-specific)
          "Avg_GPA": 3.8, // V1 field (school average)
          "Avg_SAT": 1400.0, // V1 field (school average)
          "Avg_Net_Price": 25000.0, // V1 field
          "Admission_Rate": 0.25, // V1 field
          "Admission_Statistics": [ ... V1 structure ... ],
          // ... other V1 fields like Fortune500_Hirers, Enrollment, Location, Lat/Lon, Demographics ...
          // ... potentially new V2 fields like S_academic_score, S_program_outcome_score ...
        }
        ```

## 5. API Endpoints (V2)

*   **`POST /v2/recommendations`**:
    *   Request Body: As defined in section 4.1.
    *   Response Body: List of recommendation objects (see section 4.4 Final Output Structure for frontend compatibility).
*   **`GET /v2/programs`**: Returns the list of valid, canonical program names for frontend dropdowns. Can be enhanced with aggregate data per program (e.g., number of schools offering it, average salary across all schools for that program).
*   **`GET /v2/schools/{school_name_query}/details`**: Provides comprehensive details for a school. Should also maintain V1 field structures where applicable, and add V2-specific information like how it performs on various scoring factors, and list its programs with their V2 outcome scores.

## 6. Addressing Previous Limitations

| Limitation (V1)                               | How V2 Addresses It                                                                                                                                                                                                                                                                                                                                                       |
| :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Rigidity of Rule-Based System               | Moves to a flexible multi-factor scoring system. Thresholds primarily used for normalization or preference mapping, not solely for hard tiering.                                                                                                                                                                                                                           |
| Scalability of Data Loading                 | (Acknowledged as a backend concern for future) Plan focuses on algorithmic improvement with current data loading. Database backend remains a future consideration for scaling.                                                                                                                                                                                              |
| Program-Specific Data in Recommendations    | **Core focus:** The `S_program_outcome` is central and uses program-specific earnings. Recommendations are explicitly for (school, program) pairs. `Median_Earnings_1yr/5yr` in output will be program-specific.                                                                                                                                                             |
| Tier Definitions & Variety                  | New, more descriptive tier labels based on composite scores and dominant factors. "Why" snippets provide context. Admission rate logic is integrated into the scoring and preference system rather than just a simple override.                                                                                                                                         |
| Data Freshness                              | (Operational concern) The plan relies on existing CSV update mechanisms.                                                                                                                                                                                                                                                                                                    |
| Lack of "Why"                               | "Why this school?" snippets are a planned output, derived from the score composition.                                                                                                                                                                                                                                                                                       |
| Cold Start (New Schools/Programs)           | If a new school/program is added to the data sources and fits the data schema, it will be included in processing without needing historical interaction data.                                                                                                                                                                                                              |
| Program Name Matching                       | **Primary approach expects validated program names from frontend.** Contingency for fuzzy matching (if free-text input is allowed) is more robust (`thefuzz`) than V1's length-based fallback, but carries risks if not carefully tuned.                                                                                                                                   |
| Ranking only by earnings after academic fit | The new composite score considers multiple factors simultaneously, with weights dynamically adjustable by user preferences. Program-specific earnings are a key component of the `S_program_outcome` score, which is one of many factors.                                                                                                                                     |

## 7. Frontend Compatibility Strategy

*   **Maintain Existing Fields:** For any data field currently used by the V1 frontend from the `/recommendations` or school details endpoints, the V2 endpoints will ensure these fields are present in the response with the **same name and data structure** (e.g., `Admission_Statistics` list of dicts).
*   **Additive Changes:** New V2 fields (e.g., `Composite_Score`, `V2_Tier_Label`, `Why_This_School_Snippet`, `Program_Name` alongside `School`) will be *added* to the response. Existing frontend components will ignore these new fields until they are updated to use them.
*   **Tier Label Transition:** The `Recommendation_Tier` field from V1 can either:
    1.  Be populated with the new V2 descriptive tier labels. (Simpler, but frontend needs to be okay with new string values).
    2.  A new field `V2_Recommendation_Tier` can be introduced, and the old `Recommendation_Tier` can be populated with a best-effort mapping from V2 scores to V1-like tiers for a transition period, or be deprecated. (More complex, but safer for immediate frontend compatibility if V1 tiers are hardcoded).
    *Decision on tier field needs discussion with frontend.*
*   **Communication:** Clear communication with the frontend team about new fields and any changes in the *values* of existing fields (like `Recommendation_Tier`) will be essential.

## 8. Phase 1 Implementation Notes & Skeletons (Implemented)

This section outlines the initial file skeletons and setup for V2 that have been implemented.

*   **New Service File:** `services/recommendation_service_v2.py`
    *   **Created and Implemented with Initial Logic:**
        *   `initialize_v2_data()`: Function to receive V1 DataFrames and `STATE_TO_REGION_MAP`.
        *   `STATE_TO_REGION_MAP`: Populated with example US state-to-region mappings.
        *   `SCHOOL_SIZE_THRESHOLDS`: Defined for "small", "medium", "large" categories.
        *   Helper: `normalize_min_max()` implemented for 0-1 scaling, handling NaNs and edge cases.
        *   Helper: `get_school_size_category()` implemented.
        *   **Sub-score calculators (Initial Implementation):**
            *   `calculate_academic_fit_score()`: Implemented with logic for "match", "challenge_me", "less_stress" academic focuses, handling GPA and SAT differences and averaging them. Clips score to 0-1.
            *   `calculate_program_outcome_score()`: Implemented to use normalized 1yr and 5yr program-specific earnings. Weights 5yr earnings more. Handles missing data.
            *   `calculate_affordability_score()`: Implemented. If student specifies max price, scores based on that. Otherwise, uses global normalization (lower price = higher score).
            *   `calculate_location_score()`: Implemented. Scores based on state and/or region match, using `STATE_TO_REGION_MAP`.
            *   `calculate_selectivity_score()`: Implemented. Score depends on `selectivity_preference` ("any", "moderate", "high_challenge") and school's acceptance rate.
            *   `calculate_environment_score()`: Implemented based on school type and derived size category against user preferences.
            *   `calculate_career_opportunities_score()`: Implemented based on count of Fortune 500 hirers, normalized against max hirers in candidate set.
        *   **Weighting:** `determine_composite_weights()`: Implemented with example preference adjustments for cost and career outcomes, and re-normalization of weights.
        *   **Main Orchestrator:** `recommend_schools_v2()`:
            *   Implemented program matching (exact primary, optional fuzzy via `thefuzz`).
            *   Merges candidate schools with `DF_COLLEGES_V2` and program-specific earnings from `DF_PROGRAMS_V2`.
            *   Calculates and stores all sub-scores and the composite score for each candidate.
            *   Sorts by composite score and returns top N.
            *   Includes placeholder for "Why this school?" and initial V2 tier assignment.
*   **Modifications to `app.py`:**
    *   **Imported V2 service and helpers:** `recommend_schools_v2`, `initialize_v2_data`, `STATE_TO_REGION_MAP` from `services.recommendation_service_v2`, and `act_to_sat` from `services.act_sat_conversion`.
    *   **V2 Data Initialization:** Implemented `@app.on_event("startup")` to call `initialize_v2_data`, passing the V1 global DataFrames and the `STATE_TO_REGION_MAP`.
    *   **New Pydantic Models for V2:** Defined as per plan (`StudentProfileV2`, `LocationPreferencesV2` (setting preference removed), `CostPreferencesV2`, `PreferencesV2`, `RecommendationRequestV2`, `RecommendationV2`, `RecommendationResponseV2`). `RecommendationV2` includes V1-compatible fields.
    *   **New Endpoint:** `@app.post("/v2/recommendations", response_model=RecommendationResponseV2)` created.
        *   Implements ACT to SAT conversion using `act_to_sat`.
        *   Calls `recommend_schools_v2()`.
        *   Includes initial logic for mapping the resulting DataFrame columns to `RecommendationV2` Pydantic models, including cleaning with `clean_for_json`.
    *   **No Overwriting:** Existing V1 endpoints and models remain untouched.
*   **New Test File:** `tests/test_recommendation_service_v2.py`
    *   Created with initial unit tests for:
        *   Helper functions: `normalize_min_max`, `get_school_size_category`.
        *   Sub-score functions: Basic test cases for `calculate_academic_fit_score`, `calculate_program_outcome_score`, `calculate_affordability_score`, `calculate_location_score`, `calculate_selectivity_score`, `calculate_environment_score`, `calculate_career_opportunities_score` using a small mock dataset fixture (`setup_v2_test_data`).
        *   Weighting: `determine_composite_weights`.
    *   These tests verify basic functionality and edge cases for the core scoring components.
*   **Immediate Next Steps for V2 Refinement & Completion:**
    1.  **Thoroughly Test `recommend_schools_v2` End-to-End:** Create more comprehensive integration tests for the main V2 recommendation logic, covering various student profiles and preferences.
    2.  **Refine Sub-Score Logic & Normalization:** Review and tune the placeholder logic and scaling in each `calculate_..._score()` function based on data distributions and desired sensitivity. Specifically, `calculate_program_outcome_score` and `calculate_career_opportunities_score` normalization needs to be robust using the correct series for min/max context.
    3.  **Weight Tuning:** Experiment with `base_weights` in `recommend_schools_v2` and the multipliers in `determine_composite_weights` to achieve desired recommendation characteristics.
    4.  **"Why this school?" Implementation:** Develop more sophisticated logic to generate these snippets based on the highest contributing factors to the `composite_score` relative to user preferences.
    5.  **Data Mapping in `app.py` (`/v2/recommendations`):** Solidify the mapping from the DataFrame output of `recommend_schools_v2` to the `RecommendationV2` Pydantic model. Ensure all V1-compatible fields (like `Admission_Statistics`, `Fortune500_Hirers` if computed by V2 service now) are correctly populated. Decide on the strategy for `Recommendation_Tier` vs `V2_Recommendation_Tier`.
    6.  **ACT to SAT Conversion:** Ensure the `act_to_sat` import and usage in `app.py` is correctly placed or handled within the V2 service if `effective_sat` is not passed directly.
    7.  **Error Handling & Edge Cases:** Add more robust error handling throughout the V2 service and endpoint.
    8.  **`total_schools_considered` in `RecommendationResponseV2`:** Ensure the V2 service provides this count accurately.

## 9. Future Considerations / Advanced Features

*   **Database Backend:** Transition from CSVs to a proper database (e.g., PostgreSQL, MySQL, or a NoSQL option) for improved scalability, data management, and querying capabilities.
*   **User Accounts & Saved Preferences:** Allow users to save profiles and preferences.
*   **Feedback Loop:** Enable users to rate or give feedback on recommendations, which can be used to tune the model or weights over time (potentially with ML).
*   **Machine Learning:**
    *   **Weight Optimization:** Use ML to learn optimal default weights for sub-scores based on user feedback or desired outcomes.
    *   **Collaborative Filtering:** "Students with similar profiles/preferences also liked/succeeded at these schools/programs."
    *   **Content-Based Filtering:** More advanced matching of student qualitative input (e.g., interests, career goals) to rich descriptions of schools and programs.
*   **Integration of More Data Sources:** University rankings, student reviews, more detailed course information, campus life details, more granular location attributes (urban/suburban/rural tags for schools).
*   **Advanced Location Features:** Proximity search (requires student's current location), commute considerations.
*   **Dynamic "Why":** More sophisticated natural language generation for explanations.

This V2 plan aims for a significant step up in recommendation quality and user value. Implementation will be iterative, prioritizing core scoring enhancements and frontend compatibility. 