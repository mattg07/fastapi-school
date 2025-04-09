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

def display_recommendations(data):
    # Prepare data for tabulation
    table_data = []
    for school in data['recommendations']:
        # Get most recent year's admission stats if available
        recent_stats = school.get('Admission_Statistics', [{}])[0] if school.get('Admission_Statistics') else {}
        recent_metrics = recent_stats.get('metrics', {}) if recent_stats else {}
        
        row = [
            school['School'],
            f"${school.get('Median_Earnings_1yr', 'N/A')}",
            f"${school.get('Median_Earnings_5yr', 'N/A')}",
            f"{school.get('Avg_GPA', 'N/A'):.2f}" if school.get('Avg_GPA') is not None else 'N/A',
            str(school.get('Avg_SAT', 'N/A')),
            f"{school.get('Admission_Rate', 0)*100:.1f}%" if school.get('Admission_Rate') else 'N/A',
            f"${school.get('Avg_Net_Price', 'N/A')}" if school.get('Avg_Net_Price') else 'N/A',
            str(school.get('Total_Enrollment', 'N/A')),
            len(school.get('Fortune500_Hirers', [])),
            str(school.get('Undergraduate_Enrollment', 'N/A')),
            f"{school.get('White_Enrollment_Percent', 0)*100:.1f}%" if school.get('White_Enrollment_Percent') else 'N/A',
            f"{school.get('Black_Enrollment_Percent', 0)*100:.1f}%" if school.get('Black_Enrollment_Percent') else 'N/A',
            f"{school.get('Hispanic_Enrollment_Percent', 0)*100:.1f}%" if school.get('Hispanic_Enrollment_Percent') else 'N/A',
            f"{school.get('Asian_Enrollment_Percent', 0)*100:.1f}%" if school.get('Asian_Enrollment_Percent') else 'N/A',
            # Add most recent year's admission stats
            recent_stats.get('year', 'N/A'),
            f"{recent_metrics.get('percent_applicants_admitted', 0)*100:.1f}%" if recent_metrics.get('percent_applicants_admitted') else 'N/A'
        ]
        table_data.append(row)
    
    # Print the table
    headers = [
        'School', '1yr $', '5yr $', 'GPA', 'SAT', 'Admit%', 'Net$', 'Total', 'F500', 
        'UG', 'White%', 'Black%', 'Hisp%', 'Asian%', 'Year', 'Admit%'
    ]
    print("\n=== School Recommendations ===")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def test_get_recommendations():
    test_data = {
        "gpa": 3.8,
        "sat": 1450,
        "program": "Computer Science"
    }
    response = requests.post("/recommendations", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "timestamp" in data
    assert "total_schools" in data

def test_get_random_recommendation():
    print("\nTesting random recommendation endpoint...")
    test_data = {
        "gpa": 3.8,
        "sat": 1450,
        "program": "Computer Science",
        "act": 36,
        "location_preference": "Any",
        "cost_preference": "Any",
        "admission_rate_preference": "Any",
        "salary_preference": "Any",
        "fortune500_preference": "Any",
        "number_of_recommendations": 10
    }
    
    # Make the API request
    response = requests.post(
        "http://127.0.0.1:8000/random-recommendation",
        json=test_data
    )
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"\nError status code: {response.status_code}")
        print(f"Error response content: {response.text}")
        raise
        
    data = response.json()
    
    # Verify we got exactly one school
    assert len(data["recommendations"]) == 1, f"Expected 1 school, got {len(data['recommendations'])}"
    assert data["total_schools"] == 1, f"Expected total_schools to be 1, got {data['total_schools']}"
    
    # Display the single random school
    school = data["recommendations"][0]
    print("\nRandom School Recommendation:")
    print(f"School: {school['School']}")
    print(f"Tier: {school['Recommendation_Tier']}")
    print(f"GPA/SAT: {school['Avg_GPA']}/{school['Avg_SAT']}")
    
    # Handle potentially missing data
    if school['Median_Earnings_1yr'] is not None:
        print(f"1yr Earnings: ${school['Median_Earnings_1yr']:,.0f}")
    else:
        print("1yr Earnings: N/A")
        
    if school['Median_Earnings_5yr'] is not None:
        print(f"5yr Earnings: ${school['Median_Earnings_5yr']:,.0f}")
    else:
        print("5yr Earnings: N/A")
        
    if school['Admission_Rate'] is not None:
        print(f"Admission Rate: {school['Admission_Rate']:.1%}")
    else:
        print("Admission Rate: N/A")
        
    if school['Avg_Net_Price'] is not None:
        print(f"Net Price: ${school['Avg_Net_Price']:,.0f}")
    else:
        print("Net Price: N/A")
        
    if school['Fortune500_Hirers']:
        print(f"Fortune 500 Hirers: {', '.join(school['Fortune500_Hirers'][:3])}")
    else:
        print("Fortune 500 Hirers: None")

if __name__ == "__main__":
    test_get_random_recommendation() 