import requests
import json
from tabulate import tabulate
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
    
    # Additional scenarios
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
            "name": "Biology - Average GPA, Missing SAT",
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
        
        # Print all recommendations
        for i, rec in enumerate(result["recommendations"], start=1):
            print(f"{i}. {rec['School']} - {rec['Recommendation_Tier']}")
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

def format_admission_stats(stats):
    if not stats:
        return "No data"
    
    formatted = []
    for year_data in stats:
        year = year_data['year']
        metrics = year_data.get('metrics', {})
        
        formatted.append(f"\n{year}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                value = f"{value:.2f}"
            formatted.append(f"  {metric}: {value}")
    
    return "\n".join(formatted)

def display_recommendations():
    # Make the API request
    response = requests.post(
        "http://127.0.0.1:8000/recommendations",
        json={
            "gpa": 3.5,
            "sat": 1400,
            "program": "Computer Science"
        }
    )
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"\nError status code: {response.status_code}")
        print(f"Error response content: {response.text}")
        raise
        
    data = response.json()
    
    print("\nStarting API visualization...")
    print("\n=== API Response Summary ===")
    print(f"Total Schools: {data['total_schools']}")
    print(f"Timestamp: {data['timestamp']}")
    
    # Prepare data for tabulation
    table_data = []
    for school in data['recommendations']:
        row = [
            school['School'],
            school.get('1yr_Earnings', 'N/A'),
            school.get('5yr_Earnings', 'N/A'),
            school.get('Avg_GPA', 'N/A'),
            school.get('Avg_SAT', 'N/A'),
            school.get('Admission_Rate', 'N/A'),
            school.get('Net_Price', 'N/A'),
            school.get('Enrollment', 'N/A'),
            school.get('Fortune500_Hirers', 0)
        ]
        table_data.append(row)
    
    # Print the table
    print("\n=== Consider Match ===")
    print(tabulate(
        table_data,
        headers=['School', '1yr Earnings', '5yr Earnings', 'Avg GPA', 'Avg SAT', 'Admit Rate', 'Net Price', 'Enrollment', 'Fortune 500'],
        tablefmt='grid'
    ))
    
    # Print admission statistics if available
    print("\nAdmission Statistics:")
    for school in data['recommendations']:
        print(f"\n{school['School']} — {school['Recommendation_Tier']}")
        if 'Admission_Statistics' in school and school['Admission_Statistics']:
            stats = format_admission_stats(school['Admission_Statistics'])
            print(stats)
        else:
            print("No admission statistics available")

if __name__ == "__main__":
    display_recommendations() 