import json
import os
from fastapi.testclient import TestClient
from app import app # Assuming your FastAPI app instance is named 'app' in app.py
from datetime import datetime

# Get the directory of the current test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(TEST_DIR, "quality_test_results.txt")

client = TestClient(app)

# Define student profiles (GPA, SAT, ACT)
# These are examples; you should adjust ranges and add/remove profiles as needed.
STUDENT_PROFILES = {
    "weak": {"gpa": 2.0, "sat": 900, "act": 18, "description": "Low GPA, Low Test Scores"},
    "weak_medium": {"gpa": 2.7, "sat": 1050, "act": 21, "description": "Below Average GPA, Average Test Scores"},
    "medium": {"gpa": 3.3, "sat": 1200, "act": 25, "description": "Average GPA, Good Test Scores"},
    "medium_strong": {"gpa": 3.7, "sat": 1350, "act": 29, "description": "Good GPA, Strong Test Scores"},
    "strong": {"gpa": 3.9, "sat": 1500, "act": 34, "description": "Excellent GPA, Excellent Test Scores"},
}
NUMBER_OF_RECOMMENDATIONS = 3 # Number of recommendations to fetch for each profile

def get_all_programs_from_api():
    """Fetches the list of all programs from the /programs endpoint."""
    response = client.get("/programs")
    if response.status_code != 200:
        print(f"Error fetching programs: Status {response.status_code}")
        return []
    try:
        return response.json().get("programs", [])
    except json.JSONDecodeError:
        print(f"Error decoding JSON from /programs endpoint: {response.text}")
        return []

def test_generate_recommendation_quality_report():
    """
    Generates a text report of recommendations for different student profiles across all programs.
    This test's purpose is to generate output for manual review.
    """
    all_programs = get_all_programs_from_api()

    if not all_programs:
        print("No programs found via API. Skipping quality test generation.")
        return

    print(f"\nStarting generation of quality report for {len(all_programs)} programs and {len(STUDENT_PROFILES)} profiles each. Output to: {OUTPUT_FILE}")

    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write(f"Recommendation Quality Report - Generated: {datetime.now().isoformat()}\n")
            f.write(f"Student Profiles Definition: {json.dumps(STUDENT_PROFILES)}\n")
            f.write(f"Number of Recommendations Requested Per Profile: {NUMBER_OF_RECOMMENDATIONS}\n\n")

            for i, program_name in enumerate(all_programs):
                print(f"Processing program {i+1}/{len(all_programs)}: {program_name}")
                for profile_name, profile_data in STUDENT_PROFILES.items():
                    f.write(f"Program: {program_name} - Profile: {profile_name} ({profile_data['description']})\n")

                    request_payload = {
                        "gpa": profile_data["gpa"],
                        "sat": profile_data["sat"],
                        "act": profile_data["act"],
                        "program": program_name,
                        "location_preference": "any",
                        "cost_preference": "any",
                        "admission_rate_preference": "any",
                        "salary_preference": "any",
                        "number_of_recommendations": NUMBER_OF_RECOMMENDATIONS
                    }

                    try:
                        response = client.post("/recommendations", json=request_payload)
                        if response.status_code == 200:
                            recommendations = response.json().get("recommendations", [])
                            if not recommendations:
                                f.write("  No recommendations received.\n")
                            for rec in recommendations:
                                school = rec.get("School", "N/A")
                                avg_gpa = rec.get("Avg_GPA", "N/A")
                                avg_sat = rec.get("Avg_SAT", "N/A")
                                earn_1yr = rec.get("Median_Earnings_1yr", "N/A")
                                earn_5yr = rec.get("Median_Earnings_5yr", "N/A")
                                f.write(f"  {school} - SAT: {avg_sat} - GPA: {avg_gpa} - 1yr Earnings: {earn_1yr} - 5yr Earnings: {earn_5yr}\n")
                        elif response.status_code == 404:
                            f.write("  No recommendations found (404).\n")
                        else:
                            f.write(f"  API Error: Status {response.status_code} - {response.text[:100]}\n")
                    except Exception as e:
                        f.write(f"  Client-Side Exception during API call: {str(e)}\n")
                    f.write("\n") # Add a blank line after each profile's results for readability
            
        print(f"\nQuality test report successfully generated: {OUTPUT_FILE}")
    except IOError as e:
        print(f"\nError writing quality test report to {OUTPUT_FILE}: {e}")

    assert os.path.exists(OUTPUT_FILE), f"Output file {OUTPUT_FILE} was not created." 