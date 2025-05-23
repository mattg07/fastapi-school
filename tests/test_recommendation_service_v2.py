import pytest
import pandas as pd
import numpy as np
from services.recommendation_service_v2 import (
    calculate_academic_fit_score,
    calculate_program_outcome_score,
    calculate_affordability_score,
    calculate_location_score,
    calculate_selectivity_score,
    calculate_environment_score,
    calculate_career_opportunities_score,
    determine_composite_weights,
    normalize_min_max,
    get_school_size_category,
    initialize_v2_data, # To allow initializing with mock data if needed
    DF_COLLEGES_V2, DF_PROGRAMS_V2, SCHOOL_TO_COMPANIES_V2, STATE_TO_REGION_MAP
)

# --- Fixtures for Mock Data (Optional - for more complex tests) ---
@pytest.fixture
def mock_student_profile_v2():
    return {"gpa": 3.5, "effective_sat": 1300}

@pytest.fixture
def mock_preferences_v2():
    return {
        "academic_focus": "match",
        "location": {"states": ["CA"], "region": "West"},
        "cost": {"max_net_price_per_year": 30000, "importance": "high"},
        "school_size": ["medium"],
        "school_type": "public",
        "career_outcomes_importance": "high",
        "selectivity_preference": "moderate",
        "allow_fuzzy_program_match": False
    }

@pytest.fixture(scope="module")
def setup_v2_test_data():
    # Create minimal DataFrames for testing specific functions
    # This is very basic; more comprehensive mock data would be needed for end-to-end recommend_schools_v2 tests
    global DF_COLLEGES_V2, DF_PROGRAMS_V2, SCHOOL_TO_COMPANIES_V2, STATE_TO_REGION_MAP
    DF_COLLEGES_V2 = pd.DataFrame({
        'name_clean': ['school_a', 'school_b', 'school_c'],
        'name': ['School A', 'School B', 'School C'],
        'average_gpa': [3.4, 3.8, 3.0],
        'average_sat_composite': [1200, 1400, 1100],
        'average_net_price': [25000, 50000, 15000],
        'location': ['CityA, CA', 'CityB, NY', 'CityC, TX'],
        'acceptance_rate': [0.5, 0.15, 0.75],
        'type_of_institution': ['Public', 'Private not-for-profit', 'Public'],
        'number_of_students': [12000, 25000, 1000]
    })
    DF_PROGRAMS_V2 = pd.DataFrame({
        'school_name_clean': ['school_a', 'school_b', 'school_c', 'school_a'],
        'program': ['CompSci', 'CompSci', 'Biology', 'Biology'],
        'earn_mdn_1yr': [70000, 90000, 50000, 55000],
        'earn_mdn_5yr': [100000, 150000, 75000, 80000]
    })
    SCHOOL_TO_COMPANIES_V2 = {
        'school_a': {'CompanyX': 10, 'CompanyY': 5},
        'school_b': {'CompanyZ': 20}
    }
    # STATE_TO_REGION_MAP is already defined in the service module, can be used directly or overridden here
    print("V2 Test Data Initialized for module scope")
    # No need to call initialize_v2_data here as we are setting globals for tests

# --- Tests for Helper Functions ---

def test_normalize_min_max():
    series = pd.Series([10, 20, 30, 40, 50])
    normalized = normalize_min_max(series)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized.iloc[0] == 0.0
    assert normalized.iloc[4] == 1.0
    assert normalized.iloc[2] == 0.5

    normalized_smaller_better = normalize_min_max(series, smaller_is_better=True)
    assert normalized_smaller_better.iloc[0] == 1.0
    assert normalized_smaller_better.iloc[4] == 0.0
    assert normalized_smaller_better.iloc[2] == 0.5

    series_nan = pd.Series([10, np.nan, 30])
    normalized_nan = normalize_min_max(series_nan)
    assert pd.isna(normalized_nan.iloc[1]) or normalized_nan.iloc[1] == 0.5 # fillna(0.5)

    series_same = pd.Series([20, 20, 20])
    normalized_same = normalize_min_max(series_same)
    assert all(normalized_same == 0.5) # Neutral if all same

def test_get_school_size_category():
    assert get_school_size_category(1000) == "small"
    assert get_school_size_category(10000) == "medium"
    assert get_school_size_category(20000) == "large"
    assert get_school_size_category(None) is None
    assert get_school_size_category(1999) == "small"
    assert get_school_size_category(2000) == "medium"
    assert get_school_size_category(14999) == "medium"
    assert get_school_size_category(15000) == "large"

# --- Tests for Sub-Score Functions (Basic Examples) ---

@pytest.mark.usefixtures("setup_v2_test_data")
def test_calculate_academic_fit_score():
    # Match focus
    assert calculate_academic_fit_score(3.5, 1300, 3.5, 1300, "match") == 1.0
    assert calculate_academic_fit_score(3.5, 1300, 3.3, 1200, "match") < 1.0 
    assert calculate_academic_fit_score(3.5, 1300, 3.7, 1400, "match") < 1.0 
    # Challenge focus
    assert calculate_academic_fit_score(3.5, 1300, 3.7, 1400, "challenge_me") > calculate_academic_fit_score(3.5, 1300, 3.5, 1300, "challenge_me")
    # Less_stress focus
    assert calculate_academic_fit_score(3.5, 1300, 3.3, 1200, "less_stress") > calculate_academic_fit_score(3.5, 1300, 3.5, 1300, "less_stress")
    # Test with NaNs
    assert 0.0 <= calculate_academic_fit_score(3.5, 1300, None, 1300, "match") <= 1.0
    assert calculate_academic_fit_score(3.5, 1300, None, None, "match") == 0.1

@pytest.mark.usefixtures("setup_v2_test_data")
def test_calculate_program_outcome_score():
    # Test data uses earn_mdn_1yr from 50k to 90k, earn_mdn_5yr from 75k to 150k
    # Normalization will be based on these ranges for the candidate set in a real scenario.
    # Here we mock the global_earn_stats based on our small test set
    candidate_earnings_1yr = DF_PROGRAMS_V2['earn_mdn_1yr']
    candidate_earnings_5yr = DF_PROGRAMS_V2['earn_mdn_5yr']

    score = calculate_program_outcome_score(70000, 100000, candidate_earnings_1yr, candidate_earnings_5yr)
    assert 0.0 <= score <= 1.0
    score_high = calculate_program_outcome_score(90000, 150000, candidate_earnings_1yr, candidate_earnings_5yr)
    assert score_high >= score
    assert calculate_program_outcome_score(None, None, candidate_earnings_1yr, candidate_earnings_5yr) == 0.1

@pytest.mark.usefixtures("setup_v2_test_data")
def test_calculate_affordability_score():
    candidate_net_prices = DF_COLLEGES_V2['average_net_price']
    # Student has budget
    assert calculate_affordability_score(15000, 30000, candidate_net_prices) > 0.5 # Well within budget
    assert calculate_affordability_score(30000, 30000, candidate_net_prices) == 0.5 # At budget
    assert calculate_affordability_score(35000, 30000, candidate_net_prices) == 0.0 # Over budget
    # Student has no budget (normalized globally)
    score_no_budget_low_price = calculate_affordability_score(15000, None, candidate_net_prices)
    score_no_budget_high_price = calculate_affordability_score(50000, None, candidate_net_prices)
    assert score_no_budget_low_price > score_no_budget_high_price
    assert calculate_affordability_score(None, 30000, candidate_net_prices) == 0.3

@pytest.mark.usefixtures("setup_v2_test_data")
def test_calculate_location_score():
    assert calculate_location_score("CA", "CityA", ["CA"], "West") == 1.0 # State and region match
    assert calculate_location_score("CA", "CityA", ["CA"], None) == 1.0    # State matches
    assert calculate_location_score("NY", "CityB", None, "Northeast") == 1.0 # Region matches
    assert calculate_location_score("TX", "CityC", ["CA"], "West") == 0.0  # No match
    assert calculate_location_score("CA", "CityA", ["NY"], "Northeast") == 0.0 # No match
    assert calculate_location_score("CA", "CityA", None, None) == 1.0 # No preference
    assert calculate_location_score("CA", "CityA", ["CA"], "Northeast") == 0.7 # State matches, region doesn't


@pytest.mark.usefixtures("setup_v2_test_data")
def test_calculate_selectivity_score():
    assert calculate_selectivity_score(0.15, "high_challenge") > calculate_selectivity_score(0.50, "high_challenge")
    assert calculate_selectivity_score(0.40, "moderate") == 1.0
    assert calculate_selectivity_score(0.05, "moderate") < 1.0
    assert calculate_selectivity_score(0.50, "any") == 1.0 # was 0.7, changed to 1.0 as it means full score for this factor
    assert calculate_selectivity_score(None, "any") == 0.5

@pytest.mark.usefixtures("setup_v2_test_data")
def test_calculate_environment_score():
    assert calculate_environment_score("Public", 12000, "Public", ["medium"]) == 1.0
    assert calculate_environment_score("Public", 1000, "Public", ["medium"]) == 0.5 # Size mismatch
    assert calculate_environment_score("Private not-for-profit", 12000, "Public", ["medium"]) == 0.5 # Type mismatch
    assert calculate_environment_score("Public", 12000, None, None) == 1.0 # No preference
    assert calculate_environment_score("Public", 12000, "Public", None) == 1.0 # Type pref matches, no size pref

@pytest.mark.usefixtures("setup_v2_test_data")
def test_calculate_career_opportunities_score():
    counts = pd.Series([len(SCHOOL_TO_COMPANIES_V2.get(sc, {})) for sc in DF_COLLEGES_V2['name_clean']])
    score_a = calculate_career_opportunities_score('school_a', counts)
    score_b = calculate_career_opportunities_score('school_b', counts)
    score_c = calculate_career_opportunities_score('school_c', counts) # school_c has no companies in mock
    assert score_a > score_b # School A has 2, B has 1
    assert score_b > score_c
    assert score_c == 0.1 # Low score due to no data


def test_determine_composite_weights(mock_preferences_v2):
    base_weights = {
        "academic": 0.25, "program_outcome": 0.30, "affordability": 0.15,
        "location": 0.10, "selectivity": 0.05, "environment": 0.05, "career_ops": 0.10
    }
    adj_weights = determine_composite_weights(base_weights, mock_preferences_v2)
    assert abs(sum(adj_weights.values()) - 1.0) < 1e-9 # Should sum to 1
    assert adj_weights["affordability"] > base_weights["affordability"] # Cost importance was high
    assert adj_weights["program_outcome"] > base_weights["program_outcome"] # Career importance high
    assert adj_weights["career_ops"] > base_weights["career_ops"] # Career importance high

# More comprehensive tests for recommend_schools_v2 would require more elaborate
# mocking of DataFrames or a dedicated test data setup. These are more like integration tests. 