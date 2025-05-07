import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data

def test_get_programs():
    response = client.get("/programs")
    assert response.status_code == 200
    data = response.json()
    assert "programs" in data
    assert "total_programs" in data

def test_get_recommendations():
    test_data = {
        "gpa": 3.8,
        "sat": 1450,
        "program": "Computer Science",
        "act": 32,
        "location_preference": "any",
        "cost_preference": "any",
        "admission_rate_preference": "any",
        "salary_preference": "any",
        "fortune500_preference": "any",
        "number_of_recommendations": 10
    }
    response = client.post("/recommendations", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "timestamp" in data
    assert "total_schools" in data

def test_get_program_coverage():
    response = client.get("/program_coverage")
    assert response.status_code == 200
    data = response.json()
    assert "total_programs" in data
    assert "coverage" in data
    assert "zero_match_programs" in data
    assert "num_zero_match" in data

def test_invalid_program():
    test_data = {
        "gpa": 3.8,
        "sat": 1450,
        "program": "Invalid Program Name"
    }
    response = client.post("/recommendations", json=test_data)
    assert response.status_code == 404

def test_invalid_gpa():
    test_data = {
        "gpa": 5.0,  # Invalid GPA
        "sat": 1450,
        "program": "Computer Science",
        "act": 32,
        "location_preference": "any",
        "cost_preference": "any",
        "admission_rate_preference": "any",
        "salary_preference": "any",
        "fortune500_preference": "any",
        "number_of_recommendations": 10
    }
    response = client.post("/recommendations", json=test_data)
    assert response.status_code == 422  # Validation error 