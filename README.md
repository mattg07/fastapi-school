# University Recommendation API

A FastAPI-based service that provides university recommendations based on academic performance, program of interest, and various preferences. It also offers detailed statistics for specific schools.

## Key Features

-   **Personalized University Recommendations**: Based on GPA, SAT/ACT scores, program of interest, and user preferences (location, cost, admission rate, salary).
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
│   ├── recommendation_service.py     # Core recommendation and statistics logic
│   └── act_sat_conversion.py         # Utility for ACT/SAT score conversion
├── recommendation-algo-2/            # Data files (CSV)
│   ├── colleges_data_cleaned.csv
│   ├── programs_cleaned.csv
│   ├── school_sup_data_cleaned.csv
│   ├── companies_data_cleaned.csv
│   ├── admission_trends_cleaned.csv
│   └── school_plus_image_data_cleaned.csv # Image URLs
├── tests/                            # Pytest test suite
│   ├── test_app.py                   # API endpoint tests
│   ├── test_quality_recommendations.py # Script to generate a text report for recommendation quality review
│   └── ...                           # Other test files (e.g., stat_test.py)
├── Dockerfile                        # Docker configuration
├── docker-compose.yml                # Docker Compose configuration
├── requirements.txt                  # Python dependencies
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

-   `GET /`: API information.
-   `GET /health`: Health check.
-   `GET /programs`: Lists all available academic programs.
-   `POST /recommendations`: Generates university recommendations.
    -   **Body**: `RecommendationRequest` model.
-   `GET /school/{school_name_query}`: Provides comprehensive statistics for the queried school (including image URLs).
    -   **Path Parameter**: `school_name_query`.
    -   **Response**: `SchoolStatsResponse` model.
-   `GET /school/{school_name_query}/academics`: Provides academic stats (GPA, SAT, admission rate) and image URLs.
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

Access interactive API documentation:
-   Swagger UI: `http://localhost:8000/docs`
-   ReDoc: `http://localhost:8000/redoc`

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

## Example: Get School Statistics

```bash
curl -X GET "http://localhost:8000/school/University%20of%20Pennsylvania"
```

## Example: Get Recommendations

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

## Contributing

Contributions are welcome! Please follow the standard fork, branch, commit, and pull request workflow.

## License

MIT License 