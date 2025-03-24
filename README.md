# University Recommendation API

This API provides university recommendations based on a student's GPA, SAT scores, and program of interest.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure all the required CSV files are in the `recommendation-algo-2` directory:
- colleges_data_cleaned.csv
- programs_cleaned.csv
- school_sup_data.csv
- companies_data_cleaned.csv

3. Run the API server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
Root endpoint that provides basic API information.

### POST /recommendations
Generate university recommendations based on the provided criteria.

Request body:
```json
{
    "gpa": 3.6,
    "sat": 1300,
    "program": "Computer Science"
}
```

Response:
```json
{
    "recommendations": [
        {
            "School": "University Name",
            "Recommendation_Tier": "Strong Match",
            "Has_Salary_Data": true,
            "Median_Earnings_1yr": 65000,
            "Median_Earnings_5yr": 85000,
            "Avg_GPA": 3.7,
            "Avg_SAT": 1350,
            "Fortune500_Hirers": ["Company1", "Company2"],
            "Total_Enrollment": 30000,
            "Admission_Rate": 0.15,
            "Avg_Net_Price": 25000,
            "Latitude": 40.7128,
            "Longitude": -74.0060
        }
    ],
    "timestamp": "2024-03-24 20:00:00",
    "total_schools": 1
}
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 