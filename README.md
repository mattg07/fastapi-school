# University Recommendation API

A FastAPI-based service that provides university recommendations based on academic performance, program of interest, and various preferences. It also offers detailed statistics for specific schools.

## Key Features

-   **Personalized University Recommendations**:
    -   **V1 (`/recommendations`)**: Based on GPA, SAT/ACT scores, program of interest, and basic user preferences.
    -   **V2 (`/v2/recommendations`)**: Advanced multi-factor scoring system considering academic fit, program-specific outcomes (earnings), affordability, location, school selectivity, environment, and career opportunities. Highly personalized based on detailed user preferences and provides qualitative explanations ("Why this school?").
-   **Detailed School Statistics**: Retrieve comprehensive information for a specific school, including:
    -   General academic data (average GPA, SAT, admission rates).
    -   Demographics and enrollment figures.
    -   Average 1-year and 5-year median salaries calculated across all its programs.
    -   List of programs offered with their individual salary data.
    -   Historical admission trends.
    -   Fortune 500 company hiring information.
-   **Program Information**: List available academic programs and their coverage across schools.
-   **Health Check**: Endpoint for monitoring API status.
-   **Interactive API Docs**: Automatically generated via Swagger UI and ReDoc.

## Project Structure

```
university-recommendation-server/
├── app.py                            # Main FastAPI application
├── services/                         # Business logic
│   ├── recommendation_service.py     # Core V1 recommendation and statistics logic
│   ├── recommendation_service_v2.py  # V2 multi-factor recommendation logic
│   └── act_sat_conversion.py         # Utility for ACT/SAT score conversion
├── recommendation-algo-2/            # Data files (CSV)
│   ├── colleges_data_cleaned.csv
│   ├── programs_cleaned.csv
│   ├── school_sup_data_cleaned.csv
│   ├── companies_data_cleaned.csv
│   ├── admission_trends_cleaned.csv
│   └── school_plus_image_data_cleaned.csv # Image URLs
├── tests/                            # Pytest test suite
│   ├── test_app.py                   # API endpoint tests (covers V1 and can be extended for V2)
│   ├── test_recommendation_service_v2.py # Unit tests for V2 scoring logic
│   ├── test_quality_recommendations.py # Script to generate a text report for recommendation quality review
│   └── ...                           # Other test files (e.g., stat_test.py)
├── Dockerfile                        # Docker configuration
├── docker-compose.yml                # Docker Compose configuration
├── requirements.txt                  # Python dependencies
├── plan_v2_recommendations.md        # Detailed plan for the V2 recommendation system
└── README.md                         # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd university-recommendation-server
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # For testing, ensure pytest and httpx are installed:
    # pip install pytest httpx
    ```
4.  **Data Files:** Ensure your CSV data files are present in the `recommendation-algo-2/` directory, including:
    - `colleges_data_cleaned.csv`
    - `programs_cleaned.csv`
    - `school_sup_data_cleaned.csv`
    - `companies_data_cleaned.csv`
    - `admission_trends_cleaned.csv`
    - `school_plus_image_data_cleaned.csv`
5.  **(Optional) Environment Variables:** Configure using a `.env` file if needed (e.g., for `CORS_ORIGINS` if not allowing all).

## Running the Application

### Development

```bash
uvicorn app:app --reload
```
The API will be available at `http://localhost:8000`.

### Production (with Docker)

```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

## API Endpoints

Access interactive API documentation for all endpoints:
-   Swagger UI: `http://localhost:8000/docs`
-   ReDoc: `http://localhost:8000/redoc`

### Version 1 Endpoints (Stable)

-   `GET /`: API information.
-   `GET /health`: Health check.
-   `GET /programs`: Lists all available academic programs.
-   `POST /recommendations`: Generates V1 university recommendations.
    -   **Body**: `RecommendationRequest` model.
    -   **Response**: `RecommendationResponse` model.
-   `GET /school/{school_name_query}`: Provides comprehensive statistics for the queried school.
    -   **Response**: `SchoolStatsResponse` model.
-   `GET /school/{school_name_query}/academics`: Provides academic stats and image URLs.
    -   **Response**: `SchoolAcademicStatsResponse` model.
-   `GET /school/{school_name_query}/salary`: Provides average program salary stats.
    -   **Response**: `SchoolSalaryStatsResponse` model.
-   `GET /school/{school_name_query}/demographics`: Provides enrollment and demographic stats.
    -   **Response**: `SchoolDemographicStatsResponse` model.
-   `GET /school/{school_name_query}/admission_trends`: Provides historical admission stats.
    -   **Response**: `SchoolAdmissionTrendsResponse` model.
-   `GET /school/{school_name_query}/hirers`: Provides Fortune 500 hirer stats.
    -   **Response**: `SchoolHirerStatsResponse` model.
-   `GET /program_coverage`: Returns a count of how many schools match each program.

### Version 2 Endpoints (New & Under Development)

-   **`POST /v2/recommendations`**: Generates advanced V2 university recommendations using a multi-factor scoring and personalization engine.
    -   **Request Body**: `RecommendationRequestV2` model (see `app.py` or API docs for details on `student_profile` and `preferences` structure).
    -   **Response Body**: `RecommendationResponseV2` model, including composite scores, new tier labels, "Why this school?" explanations, individual sub-scores, and V1-compatible fields.
    -   *Note: Full capabilities of preferences (e.g., region, school size/type) are dependent on data richness and ongoing tuning of the V2 algorithm as detailed in `plan_v2_recommendations.md`.*

-   *(Future V2 Endpoints - as per plan_v2_recommendations.md)*
    -   *`GET /v2/programs` (potentially with enhanced program info)*
    -   *`GET /v2/schools/{school_name_query}/details` (with V2 scoring factor details)*

## Testing

The project uses `pytest` for testing.

1.  **Run all tests:**
    ```bash
    pytest
    ```
    This will execute:
    -   Unit and integration tests for API endpoints (`tests/test_app.py`).
    -   Other specialized tests (e.g., `stat_test.py`).
    -   The recommendation quality report generation.

2.  **Recommendation Quality Report:**
    The test `tests/test_quality_recommendations.py::test_generate_recommendation_quality_report` (run as part of the normal `pytest` command) generates a text file: `tests/quality_test_results.txt`.
    This file contains recommendations for various student profiles across all programs, formatted for manual review.

## Example: Get School Statistics (V1)

```bash
curl -X GET "http://localhost:8000/school/University%20of%20Pennsylvania"
```

## Example: Get Recommendations (V1)

```bash
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
           "gpa": 3.7,
           "sat": 1400,
           "program": "Computer Science",
           "act": 30,
           "number_of_recommendations": 3
         }'
```

## Example: Get Recommendations (V2 - New)

```bash
curl -X POST "http://localhost:8000/v2/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
           "student_profile": {
             "gpa": 3.7,
             "sat": 1350,
             "act": 29
           },
           "program_query": "Computer Science",
           "preferences": {
             "academic_focus": "match",
             "location": {"states": ["CA"], "region": "West"},
             "cost": {"max_net_price_per_year": 40000, "importance": "high"},
             "school_size": ["medium", "large"],
             "school_type": "public",
             "career_outcomes_importance": "high",
             "selectivity_preference": "moderate"
           },
           "number_of_recommendations": 5
         }'
```

## Contributing

Contributions are welcome! Please follow the standard fork, branch, commit, and pull request workflow.

## License

MIT License 