import requests
import json
from datetime import datetime

def get_valid_programs():
    """Get the list of valid program names from the API."""
    response = requests.get("http://localhost:8000/programs")
    response.raise_for_status()
    return response.json()["programs"]

def test_recommendations():
    # API endpoint
    url = "http://localhost:8000/recommendations"
    
    # Get valid program names
    try:
        valid_programs = get_valid_programs()
        print(f"\nFound {len(valid_programs)} valid programs")
        print("Sample programs:")
        for prog in valid_programs[:5]:
            print(f"- {prog}")
    except Exception as e:
        print(f"Error getting valid programs: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "name": "Computer Science - High GPA/SAT",
            "data": {
                "gpa": 3.8,
                "sat": 1450,
                "program": "Computer Science",
                "act": 32,                             # ADDED
                "location_preference": "any",          # ADDED
                "cost_preference": "any",              # ADDED
                "admission_rate_preference": "any",    # ADDED
                "salary_preference": "any",            # ADDED
                "fortune500_preference": "any",        # ADDED
                "number_of_recommendations": 10        # ADDED
            }
        },
        {
            "name": "Biology - Medium GPA/SAT",
            "data": {
                "gpa": 3.5,
                "sat": 1300,
                "program": "Biology, General",
                "act": 28,
                "location_preference": "any",
                "cost_preference": "any",
                "admission_rate_preference": "any",
                "salary_preference": "any",
                "fortune500_preference": "any",
                "number_of_recommendations": 10
            }
        },
        {
            "name": "Business - Lower GPA/SAT",
            "data": {
                "gpa": 3.2,
                "sat": 1200,
                "program": "Business Administration and Management, General",
                "act": 25,
                "location_preference": "any",
                "cost_preference": "any",
                "admission_rate_preference": "any",
                "salary_preference": "any",
                "fortune500_preference": "any",
                "number_of_recommendations": 10
            }
        }
    ]
    
    # Additional scenarios
    additional_test_cases = [
        {
            "name": "Business - Low GPA, High SAT",
            "data": {
                "gpa": 2.8,
                "sat": 1450,
                "program": "Business Administration and Management, General",
                "act": 24,
                "location_preference": "any",
                "cost_preference": "any",
                "admission_rate_preference": "any",
                "salary_preference": "any",
                "fortune500_preference": "any",
                "number_of_recommendations": 10
            }
        },
        {
            "name": "Computer Science - High GPA, Low SAT",
            "data": {
                "gpa": 3.9,
                "sat": 1100,
                "program": "Computer Science",
                "act": 30,
                "location_preference": "any",
                "cost_preference": "any",
                "admission_rate_preference": "any",
                "salary_preference": "any",
                "fortune500_preference": "any",
                "number_of_recommendations": 10
            }
        },
        {
            "name": "Agricultural Business - Rare Major, Strong Student",
            "data": {
                "gpa": 3.95,
                "sat": 1550,
                "program": "Agricultural Business and Management",
                "act": 33,
                "location_preference": "any",
                "cost_preference": "any",
                "admission_rate_preference": "any",
                "salary_preference": "any",
                "fortune500_preference": "any",
                "number_of_recommendations": 10
            }
        },
        {
            "name": "Biology - Average GPA, Missing SAT",
            "data": {
                "gpa": 3.0,
                "sat": 0,   # or maybe 900 to represent a lower-than-average
                "program": "Biology, General",
                "act": 20,
                "location_preference": "any",
                "cost_preference": "any",
                "admission_rate_preference": "any",
                "salary_preference": "any",
                "fortune500_preference": "any",
                "number_of_recommendations": 10
            }
        },
        {
            "name": "Computer Science - Perfect Stats",
            "data": {
                "gpa": 4.0,
                "sat": 1580,
                "program": "Computer Science",
                "act": 36,
                "location_preference": "any",
                "cost_preference": "any",
                "admission_rate_preference": "any",
                "salary_preference": "any",
                "fortune500_preference": "any",
                "number_of_recommendations": 10
            }
        }
    ]
    
    test_cases.extend(additional_test_cases)

    # Run tests
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        print("-" * 50)
        
        try:
            response = requests.post(url, json=test['data'])
            response.raise_for_status()  # Raise exception if status != 2xx
            
            result = response.json()
            print(f"Status Code: {response.status_code}")
            print(f"Total Schools Found: {result['total_schools']}")
            print(f"Timestamp: {result['timestamp']}")
            
            # Print top 3 recommendations
            print("\nTop 3 Recommendations:")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"\n{i}. {rec['School']}")
                print(f"   Tier: {rec['Recommendation_Tier']}")
                print(f"   GPA/SAT: {rec['Avg_GPA']}/{rec['Avg_SAT']}")
                # Checking some fields:
                print(f"   Admission Rate: {rec.get('Admission_Rate')}")
                print(f"   Fortune 500 Hirers: {rec.get('Fortune500_Hirers')[:3]}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
        except Exception as e:
            print(f"Processing Error: {e}")

def test_additional_scenario():
    """
    Tests fallback logic for a rare program by requesting medium/high stats
    that might yield fewer than 10 strong matches.
    """
    url = "http://localhost:8000/recommendations"
    
    scenario = {
        "name": "Aerospace - Medium/High Fallback Check",
        "data": {
            "gpa": 3.6,
            "sat": 1350,
            "program": "Aerospace Aeronautical and Astronautical Engineering",
            "act": 29,
            "location_preference": "any",
            "cost_preference": "any",
            "admission_rate_preference": "any",
            "salary_preference": "any",
            "fortune500_preference": "any",
            "number_of_recommendations": 10
        }
    }
    
    print(f"\nRunning Additional Test: {scenario['name']}")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=scenario["data"])
        response.raise_for_status()
        
        result = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Total Schools Found: {result['total_schools']}")
        print(f"Timestamp: {result['timestamp']}")
        
        # Print all recommendations
        for i, rec in enumerate(result["recommendations"], start=1):
            print(f"{i}. {rec['School']} - {rec['Recommendation_Tier']}")
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    print("Starting API tests...")
    test_recommendations()
    test_additional_scenario()