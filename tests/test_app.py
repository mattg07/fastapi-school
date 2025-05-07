from fastapi.testclient import TestClient
from app import app # Assuming your FastAPI app instance is named 'app' in app.py
import pytest # We'll need pytest to run this
from unittest.mock import patch # Added for mocking
import pandas as pd # Added for potentially returning empty DataFrame

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "University Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "recommendations": "/recommendations (POST)",
            "programs": "/programs (GET)"
        }
    }

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "healthy"
    assert "timestamp" in json_response
    assert json_response["version"] == "1.0.0"

def test_get_programs():
    response = client.get("/programs")
    assert response.status_code == 200
    json_response = response.json()
    assert "programs" in json_response
    assert "total_programs" in json_response
    assert isinstance(json_response["programs"], list)
    assert json_response["total_programs"] == len(json_response["programs"])
    # We could add more assertions here, e.g., checking if known programs exist if VALID_PROGRAMS is static

# A basic success test for the recommendations endpoint.
# This test assumes that there's some combination of inputs
# that will return a successful recommendation.
# For more robust testing, you might need to mock the
# `recommend_schools` function or set up a test database/data files.
def test_get_recommendations_success():
    # This is a sample request. You might need to adjust it
    # based on what inputs are known to produce results
    # with your current data and recommendation logic.
    request_data = {
        "gpa": 3.5,
        "sat": 1200,
        "program": "Computer Science", # Assuming this is a valid program
        "act": 25, # Added based on your RecommendationRequest model
        "location_preference": "any",
        "cost_preference": "any",
        "admission_rate_preference": "any",
        "salary_preference": "any",
        "number_of_recommendations": 3
    }
    response = client.post("/recommendations", json=request_data)

    # If you have a setup where "Computer Science" with these scores
    # might not always return results (e.g., if data is sparse),
    # this assertion might fail. You might need to use a program
    # known to have data or mock the underlying service.
    assert response.status_code == 200
    json_response = response.json()
    assert "recommendations" in json_response
    assert "timestamp" in json_response
    assert "total_schools" in json_response
    assert len(json_response["recommendations"]) <= request_data["number_of_recommendations"]
    if json_response["total_schools"] > 0:
        assert len(json_response["recommendations"]) > 0
        first_rec = json_response["recommendations"][0]
        assert "School" in first_rec
        assert "Recommendation_Tier" in first_rec
        # Add more checks for the structure of a recommendation if needed

@patch('app.recommend_schools') # Target 'recommend_schools' as imported/used in app.py
def test_get_recommendations_invalid_program(mock_recommend_schools):
    # Configure the mock to return None, simulating no recommendations found
    mock_recommend_schools.return_value = None

    request_data = {
        "gpa": 3.5,
        "sat": 1200,
        "program": "AnyProgramNameWillDoNowBecauseWeAreMocking",
        "act": 25,
        "number_of_recommendations": 3
    }
    response = client.post("/recommendations", json=request_data)

    assert response.status_code == 404
    json_response = response.json()
    assert "detail" in json_response
    # Check for the specific detail message from app.py
    assert f"No recommendations found for program: {request_data['program']}" in json_response["detail"]

# Test for /program_coverage endpoint
def test_get_program_coverage():
    response = client.get("/program_coverage")
    assert response.status_code == 200
    json_response = response.json()

    # Check if the top-level keys are present
    assert "total_programs" in json_response
    assert "coverage" in json_response
    assert "zero_match_programs" in json_response
    assert "num_zero_match" in json_response

    # Check types
    assert isinstance(json_response["total_programs"], int)
    assert isinstance(json_response["coverage"], dict)
    assert isinstance(json_response["zero_match_programs"], list)
    assert isinstance(json_response["num_zero_match"], int)

    # Basic logic checks
    assert json_response["total_programs"] >= 0
    assert json_response["num_zero_match"] == len(json_response["zero_match_programs"])

    # If there's coverage data, check its structure
    if json_response["total_programs"] > 0 and len(json_response["coverage"]) > 0:
        first_program_in_coverage = list(json_response["coverage"].keys())[0]
        assert isinstance(json_response["coverage"][first_program_in_coverage], int)
        assert json_response["coverage"][first_program_in_coverage] >= 0

    # This part of the test assumes the data files exist at the expected paths.
    # If they don't, your endpoint returns an error message.
    # You might want separate tests for the file-not-found scenario,
    # or ensure test data files are present in a test environment.

# Tests for /school/{school_name_query} endpoint

def test_get_school_stats_success():
    # NOTE: This test assumes "University of Pennsylvania" (or its standardized form)
    # exists in your data and has associated program/salary/hirer info for a meaningful test.
    # Adjust the school_name if needed, or consider mocking get_school_statistics for more controlled testing.
    school_name = "University of Pennsylvania"
    response = client.get(f"/school/{school_name}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["school_name_display"].lower() == school_name.lower() # Check display name (case-insensitive)
    assert "school_name_standardized" in data
    assert data["school_name_standardized"] is not None # Should have a standardized name
    
    # Check for some key statistical fields presence
    assert "average_gpa" in data
    assert "average_sat" in data
    assert "admission_rate" in data
    assert "total_enrollment" in data
    assert "undergraduate_enrollment" in data
    assert "avg_program_median_earnings_1yr" in data # This can be None if no programs have 1yr data
    assert "avg_program_median_earnings_5yr" in data # This can be None if no programs have 5yr data
    assert "programs_offered_count" in data
    assert data["programs_offered_count"] >= 0
    assert "program_salary_details" in data
    assert isinstance(data["program_salary_details"], list)
    assert "admission_statistics" in data # This can be None or an empty list
    assert "fortune_500_hirers" in data
    assert isinstance(data["fortune_500_hirers"], list)
    assert "data_sources_used" in data
    assert isinstance(data["data_sources_used"], list)
    assert "query_timestamp" in data

    # If programs are offered, check structure of one program detail
    if data["programs_offered_count"] > 0 and data["program_salary_details"]:
        first_program = data["program_salary_details"][0]
        assert "program_name" in first_program
        assert "median_earnings_1yr" in first_program
        assert "median_earnings_5yr" in first_program

    # If hirers are present, check structure
    if data["fortune_500_hirers"]:
        first_hirer = data["fortune_500_hirers"][0]
        assert "company_name" in first_hirer
        assert "alumni_count" in first_hirer

def test_get_school_stats_not_found():
    school_name = "This Is A Non Existent University Name Xyz123"
    response = client.get(f"/school/{school_name}")
    
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert f"School matching '{school_name}' not found" in data["detail"]

def test_get_school_stats_query_case_insensitivity():
    # Assuming "Harvard University" is in your data for this test.
    # Adjust if necessary.
    school_name_variant_case = "hArVaRd UnIvErSiTy"
    expected_display_name_segment = "Harvard University" # Actual display name might vary slightly based on your data source
    
    response = client.get(f"/school/{school_name_variant_case}")
    assert response.status_code == 200
    data = response.json()
    assert expected_display_name_segment.lower() in data["school_name_display"].lower()
    assert data["school_name_standardized"] is not None 