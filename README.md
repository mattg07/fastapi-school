# University Recommendation API

A FastAPI-based service that provides university recommendations based on GPA, SAT scores, and program of interest.

## Features

- University recommendations based on:
  - GPA
  - SAT scores
  - Program of interest
- Comprehensive school information including:
  - Institution details
  - Admission rates
  - Salary data
  - Fortune 500 hiring companies
  - Location data
  - Net price information

## Project Structure

```
university-recommendation-server/
├── app.py                 # Main FastAPI application
├── config/               # Configuration files
│   └── settings.py       # Application settings
├── services/            # Business logic
│   └── recommendation_service.py
├── data/               # Data files
│   ├── colleges_data_cleaned.csv
│   ├── programs_cleaned.csv
│   ├── school_sup_data.csv
│   └── companies_data_cleaned.csv
├── tests/              # Test files
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── requirements.txt    # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd university-recommendation-server
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your data files in the `data/` directory:
- colleges_data_cleaned.csv
- programs_cleaned.csv
- school_sup_data.csv
- companies_data_cleaned.csv

5. Create a `.env` file in the root directory:
```env
DEBUG=0
CORS_ORIGINS=http://localhost:3000
```

## Development

Run the development server:
```bash
uvicorn app:app --reload
```

## Production Deployment

### Using Docker

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`

### Without Docker

1. Set up a production server (e.g., using Gunicorn):
```bash
gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## API Endpoints

- `GET /`: API information
- `GET /health`: Health check endpoint
- `GET /programs`: List of available programs
- `POST /recommendations`: Get university recommendations
- `GET /program_coverage`: Get program coverage statistics

## Example Request

```bash
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
           "gpa": 3.8,
           "sat": 1450,
           "program": "Computer Science"
         }'
```

## Testing

Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 