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
    
    # Test cases with valid program names
    test_cases = [
        {
            "name": "Computer Science - High GPA/SAT",
            "data": {
                "gpa": 3.8,
                "sat": 1450,
                "program": "Computer Science"
            }
        },
        {
            "name": "Biology - Medium GPA/SAT",
            "data": {
                "gpa": 3.5,
                "sat": 1300,
                "program": "Biology, General"
            }
        },
        {
            "name": "Business - Lower GPA/SAT",
            "data": {
                "gpa": 3.2,
                "sat": 1200,
                "program": "Business Administration and Management, General"
            }
        }
    ]
    
    # Adding new test scenarios
    additional_test_cases = [
        {
            "name": "Business - Low GPA, High SAT",
            "data": {
                "gpa": 2.8,
                "sat": 1450,
                "program": "Business Administration and Management, General"
            }
        },
        {
            "name": "Computer Science - High GPA, Low SAT",
            "data": {
                "gpa": 3.9,
                "sat": 1100,
                "program": "Computer Science"
            }
        },
        {
            "name": "Agricultural Business - Rare Major, Strong Student",
            "data": {
                "gpa": 3.95,
                "sat": 1550,
                "program": "Agricultural Business and Management"
            }
        },
        {
            "name": "Biology - Average GPA, Missing/Low SAT",
            "data": {
                "gpa": 3.0,
                "sat": 0, 
                "program": "Biology, General"
            }
        },
        {
            "name": "Computer Science - Perfect Stats",
            "data": {
                "gpa": 4.0,
                "sat": 1580,
                "program": "Computer Science"
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
            response.raise_for_status()  # Raise an exception for bad status codes
            
            result = response.json()
            
            # Print summary
            print(f"Status Code: {response.status_code}")
            print(f"Total Schools Found: {result['total_schools']}")
            print(f"Timestamp: {result['timestamp']}")
            
            # Print top 3 recommendations
            print("\nTop 3 Recommendations:")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"\n{i}. {rec['School']}")
                print(f"   Tier: {rec['Recommendation_Tier']}")
                print(f"   GPA/SAT: {rec['Avg_GPA']}/{rec['Avg_SAT']}")
                earnings = f"${rec['Median_Earnings_1yr']:,.0f}/${rec['Median_Earnings_5yr']:,.0f}" if rec['Median_Earnings_1yr'] and rec['Median_Earnings_5yr'] else "No data"
                print(f"   Earnings (1yr/5yr): {earnings}")
                print(f"   Fortune 500 Hirers: {', '.join(rec['Fortune500_Hirers'][:3])}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
        except Exception as e:
            print(f"Error processing response: {e}")

def test_additional_scenario():
    """
    Tests the fallback logic for a relatively rare program ('Aerospace Aeronautical and Astronautical Engineering')
    by requesting medium/high stats that might yield fewer than 10 strong matches.
    """
    url = "http://localhost:8000/recommendations"
    
    scenario = {
        "name": "Aerospace - Medium/High Fallback Check",
        "data": {
            "gpa": 3.6,
            "sat": 1350,
            "program": "Aerospace Aeronautical and Astronautical Engineering"
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
        
        recommendations = result["recommendations"]
        
        # Print all recommendations to see if fallback logic was triggered
        for i, rec in enumerate(recommendations, start=1):
            school = rec["School"]
            tier = rec["Recommendation_Tier"]
            print(f"{i}. {school} - {tier}")
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    print("Starting API tests...")
    test_recommendations()
    
    # Run the additional fallback scenario test
    test_additional_scenario() 