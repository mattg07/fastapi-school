#!/bin/bash

# Create a directory for responses if it doesn't exist
mkdir -p responses

# Test /programs endpoint
echo "Testing /programs endpoint..."
curl -X GET "http://localhost:8000/programs" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  > responses/programs_response.json

# Test /recommendations endpoint with different scenarios
echo "Testing /recommendations endpoint with Computer Science - High GPA/SAT..."
curl -X POST "http://localhost:8000/recommendations" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "program": "Computer Science",
    "gpa": 3.9,
    "sat": 1513,
    "location": "Any"
  }' \
  > responses/recommendations_cs_high.json

echo "Testing /recommendations endpoint with Biology - Medium GPA/SAT..."
curl -X POST "http://localhost:8000/recommendations" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "program": "Biology",
    "gpa": 3.6,
    "sat": 1250,
    "location": "Any"
  }' \
  > responses/recommendations_bio_medium.json

echo "Testing /recommendations endpoint with Business - Lower GPA/SAT..."
curl -X POST "http://localhost:8000/recommendations" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "program": "Business",
    "gpa": 3.4,
    "sat": 1134,
    "location": "Any"
  }' \
  > responses/recommendations_business_lower.json

echo "Testing /recommendations endpoint with Aerospace Engineering - High Stats..."
curl -X POST "http://localhost:8000/recommendations" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "program": "Aerospace Aeronautical and Astronautical Engineering",
    "gpa": 3.8,
    "sat": 1450,
    "location": "Any"
  }' \
  > responses/recommendations_aero_high.json

echo "Testing /recommendations endpoint with Agricultural Business - Rare Major..."
curl -X POST "http://localhost:8000/recommendations" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "program": "Agricultural Business and Management",
    "gpa": 3.9,
    "sat": 1394,
    "location": "Any"
  }' \
  > responses/recommendations_ag_business.json

echo "All responses have been saved to the responses directory." 