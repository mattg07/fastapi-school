import requests
import json
from datetime import datetime
import pytest

API_BASE = "http://localhost:8000"

@pytest.fixture
def program_name():
    # You can customize this list or the logic to select a program name
    return "Computer Science" # Default example

def get_valid_programs():
    """Get the list of valid program names from the API."""
    response = requests.get(f"{API_BASE}/programs")
    response.raise_for_status()
    data = response.json()
    return data.get("programs", [])

def test_recommendation_with_admissions(program_name):
    """Request recommendations for a given program and print the admissions history."""
    url = f"{API_BASE}/recommendations"
    payload = {
        "gpa": 3.7,
        "sat": 1400,
        "program": program_name,
        "act": 31,
        "location_preference": "any",
        "cost_preference": "any",
        "admission_rate_preference": "any",
        "salary_preference": "any",
        "fortune500_preference": "any",
        "number_of_recommendations": 5
    }
    
    print(f"\nRequesting recommendations for: {program_name}")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    print(f"\nStatus Code: {response.status_code}")
    print(f"Timestamp: {result.get('timestamp')}")
    print(f"Total Schools Found: {result.get('total_schools')}")
    
    recommendations = result.get("recommendations", [])
    if not recommendations:
        print("No recommendations found.")
        return
    
    for idx, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {idx}: {rec.get('School')}")
        print(f"  Tier: {rec.get('Recommendation_Tier')}")
        print(f"  Avg GPA / SAT: {rec.get('Avg_GPA')} / {rec.get('Avg_SAT')}")
        print(f"  Admission Rate: {rec.get('Admission_Rate')}")
        print("  Admission Statistics:")
        admission_stats = rec.get("Admission_Statistics", [])
        if admission_stats:
            for stat in admission_stats:
                year = stat.get("year")
                metrics = stat.get("metrics")
                print(f"    Year: {year}")
                for metric, value in metrics.items():
                    print(f"      {metric}: {value}")
        else:
            print("    No admissions data available.")

if __name__ == "__main__":
    print("Starting API test for admissions history...")
    
    try:
        programs = get_valid_programs()
        print(f"Found {len(programs)} valid programs.")
        # You can choose a specific program; for example, if "Aerospace Aeronautical and Astronautical Engineering" is valid:
        chosen_program = "Aerospace Aeronautical and Astronautical Engineering" if "Aerospace Aeronautical and Astronautical Engineering" in programs else programs[0]
        
        test_recommendation_with_admissions(chosen_program)
    except Exception as e:
        print(f"Test failed: {e}")