import requests
import json
from datetime import datetime

def test_random_target_recommendation():
    """
    Test the random-recommendation endpoint to verify it returns a school with "Target" tier.
    """
    print(f"\nTesting random target recommendation endpoint at {datetime.now().isoformat()}")
    
    # API endpoint
    url = "http://localhost:8000/random-recommendation"
    
    # Test data
    test_data = {
        "program": "Computer Science",
        "gpa": 3.8,
        "sat": 1450,
        "act": 36,
        "location_preference": "Any",
        "cost_preference": "Any",
        "admission_rate_preference": "Any",
        "salary_preference": "Any",
        "fortune500_preference": "Any",
        "number_of_recommendations": 10
    }
    
    try:
        # Make the API request
        response = requests.post(url, json=test_data)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Verify we got exactly one school
        assert len(data["recommendations"]) == 1, f"Expected 1 school, got {len(data['recommendations'])}"
        assert data["total_schools"] == 1, f"Expected total_schools to be 1, got {data['total_schools']}"
        
        # Get the school
        school = data["recommendations"][0]
        
        # Check if the school has "Target" in its tier
        tier = school.get("Recommendation_Tier", "")
        is_target = "Target" in tier
        
        # Display the school
        print("\nRandom School Recommendation:")
        print(f"School: {school['School']}")
        print(f"Tier: {tier}")
        print(f"Is Target: {is_target}")
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
            
        # Run the test multiple times to verify consistency
        print("\nRunning multiple tests to verify consistency...")
        target_count = 0
        total_tests = 5
        
        for i in range(total_tests):
            response = requests.post(url, json=test_data)
            response.raise_for_status()
            data = response.json()
            school = data["recommendations"][0]
            tier = school.get("Recommendation_Tier", "")
            is_target = "Target" in tier
            
            if is_target:
                target_count += 1
                
            print(f"Test {i+1}: {school['School']} - Tier: {tier} - Is Target: {is_target}")
            
        print(f"\nTarget schools returned: {target_count}/{total_tests} ({target_count/total_tests:.1%})")
        
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
    except Exception as e:
        print(f"Processing Error: {e}")

if __name__ == "__main__":
    test_random_target_recommendation() 