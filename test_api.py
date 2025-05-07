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
            
            # Print top 3 recommendations in more detail
            print("\nDetailed Top 3 Recommendations:")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"\n{i}. School: {rec.get('School')}")
                print(f"   Tier: {rec.get('Recommendation_Tier')}")
                print(f"   Avg GPA (School): {rec.get('Avg_GPA')}")
                print(f"   Avg SAT (School): {rec.get('Avg_SAT')}")
                print(f"   Has Salary Data: {rec.get('Has_Salary_Data')}")
                if rec.get('Has_Salary_Data'):
                    print(f"     Median Earnings (1yr): ${rec.get('Median_Earnings_1yr'):,.0f}" if rec.get('Median_Earnings_1yr') else "     Median Earnings (1yr): N/A")
                    print(f"     Median Earnings (5yr): ${rec.get('Median_Earnings_5yr'):,.0f}" if rec.get('Median_Earnings_5yr') else "     Median Earnings (5yr): N/A")
                print(f"   Admission Rate: {rec.get('Admission_Rate'):.1%}" if rec.get('Admission_Rate') is not None else "   Admission Rate: N/A")
                print(f"   Total Enrollment: {rec.get('Total_Enrollment'):,}" if rec.get('Total_Enrollment') is not None else "   Total Enrollment: N/A")
                
                fortune_hirers_list = rec.get('Fortune500_Hirers', [])
                if fortune_hirers_list:
                    print(f"   Fortune 500 Hirers ({len(fortune_hirers_list)} total companies):")
                    for hirer_info in fortune_hirers_list[:3]: # Show top 3
                        company = hirer_info.get("company_name", "Unknown Company")
                        count = hirer_info.get("alumni_count", "N/A")
                        print(f"     - {company}: {count} alumni")
                    if len(fortune_hirers_list) > 3:
                        print("       ...")
                else:
                    print("   Fortune 500 Hirers: N/A")

                admission_stats = rec.get('Admission_Statistics', [])
                if admission_stats:
                    latest_stats = sorted(admission_stats, key=lambda x: x.get('year', 0), reverse=True)[0]
                    print(f"   Latest Admission Stats ({latest_stats.get('year')}):")
                    for metric, value in latest_stats.get('metrics', {}).items():
                        if isinstance(value, float) and not value.is_integer():
                            print(f"     {metric}: {value:.3f}") # More precision for floats
                        else:
                            print(f"     {metric}: {value}")
                else:
                    print("   Admission Statistics: N/A")
                
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
        
        # Print all recommendations with key details
        print("\nRecommendations Received:")
        for i, rec in enumerate(result["recommendations"], start=1):
            print(f"\n{i}. School: {rec.get('School')}")
            print(f"   Tier: {rec.get('Recommendation_Tier')}")
            print(f"   Avg GPA (School): {rec.get('Avg_GPA')}")
            print(f"   Avg SAT (School): {rec.get('Avg_SAT')}")
            if rec.get('Has_Salary_Data'):
                 print(f"     Median Earnings (1yr): ${rec.get('Median_Earnings_1yr'):,.0f}" if rec.get('Median_Earnings_1yr') else "     Median Earnings (1yr): N/A")
            print(f"   Admission Rate: {rec.get('Admission_Rate'):.1%}" if rec.get('Admission_Rate') is not None else "   Admission Rate: N/A")
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    print("Starting API tests...")
    test_recommendations()
    test_additional_scenario()