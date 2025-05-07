from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import json
import math

from services.recommendation_service import recommend_schools, VALID_PROGRAMS

import pandas as pd
import numpy as np

# =====================================================
# Setup FastAPI
# =====================================================

app = FastAPI(
    title="School Selector API",
    description="API for generating university recommendations based on GPA, SAT scores, and program of interest",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Models
# =====================================================

class RecommendationRequest(BaseModel):
    gpa: float
    sat: int
    program: str
    act: float
    location_preference: Optional[str] = "any"
    cost_preference: Optional[str] = "any"
    admission_rate_preference: Optional[str] = "any"
    salary_preference: Optional[str] = "any"
    number_of_recommendations: int

class AdmissionMetrics(BaseModel):
    metric: str
    value: float

class AdmissionYearStats(BaseModel):
    year: int
    metrics: dict

class Recommendation(BaseModel):
    School: str
    Recommendation_Tier: str
    Has_Salary_Data: bool
    Median_Earnings_1yr: Optional[float] = None
    Median_Earnings_5yr: Optional[float] = None
    Avg_GPA: Optional[float] = None
    Avg_SAT: Optional[float] = None
    Fortune500_Hirers: List[str]
    Total_Enrollment: Optional[int] = None
    Admission_Rate: Optional[float] = None
    Avg_Net_Price: Optional[float] = None
    Latitude: Optional[float] = None
    Longitude: Optional[float] = None
    Admission_Statistics: Optional[List[AdmissionYearStats]] = None
    Undergraduate_Enrollment: Optional[int] = None
    White_Enrollment_Percent: Optional[float] = None
    Black_Enrollment_Percent: Optional[float] = None
    Hispanic_Enrollment_Percent: Optional[float] = None
    Asian_Enrollment_Percent: Optional[float] = None

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    timestamp: str
    total_schools: int

# =====================================================
# Utility function to clean data for JSON
# =====================================================
def clean_for_json(obj):
    """
    Recursively convert NumPy and built-in numeric types to native Python types.
    Replace any NaN, inf, or -inf values with None.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    return obj

# =====================================================
# Endpoints
# =====================================================

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    Generates a list of recommended schools based on the user's GPA, SAT, program, etc.
    Also attaches the admission trends history to each recommendation.
    """
    try:
        df_recommendations = recommend_schools(
            program=request.program,
            gpa=request.gpa,
            sat=request.sat,
            act=request.act,
            location_preference=request.location_preference,
            cost_preference=request.cost_preference,
            admission_rate_preference=request.admission_rate_preference,
            salary_preference=request.salary_preference,
            number_of_recommendations=request.number_of_recommendations
        )
        
        if df_recommendations is None or df_recommendations.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for program: {request.program}"
            )
        
        recommendations_list = df_recommendations.to_dict(orient="records")
        recommendations_list = clean_for_json(recommendations_list)
        
        # Optional debug output
        if recommendations_list:
            print("\nDebug: First recommendation structure:")
            print(json.dumps(recommendations_list[0], indent=2))
        
        return RecommendationResponse(
            recommendations=recommendations_list,
            timestamp=datetime.now().isoformat(),
            total_schools=len(recommendations_list)
        )
        
    except Exception as e:
        import traceback
        print(f"\nError details: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}\n{traceback.format_exc()}"
        )

@app.get("/programs")
async def get_programs():
    """Get the list of valid program names."""
    return {
        "programs": VALID_PROGRAMS,
        "total_programs": len(VALID_PROGRAMS)
    }

@app.get("/")
async def root():
    return {
        "message": "University Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "recommendations": "/recommendations (POST)",
            "programs": "/programs (GET)"
        }
    }

@app.get("/program_coverage")
def get_program_coverage():
    """
    Return a count of how many schools match each program in programs_cleaned.csv.
    Great for diagnosing programs with very few or zero matches.
    """
    import pandas as pd

    path_colleges = "recommendation-algo-2/colleges_data_cleaned.csv"
    path_programs = "recommendation-algo-2/programs_cleaned.csv"
    path_school_sup = "recommendation-algo-2/school_sup_data_cleaned.csv"

    if not os.path.exists(path_colleges) or not os.path.exists(path_programs) or not os.path.exists(path_school_sup):
        return {
            "error": "One or more data files not found. Check your file paths!",
            "colleges_exists": os.path.exists(path_colleges),
            "programs_exists": os.path.exists(path_programs),
            "school_sup_exists": os.path.exists(path_school_sup)
        }

    df_colleges = pd.read_csv(path_colleges)
    df_programs = pd.read_csv(path_programs)
    df_school_sup = pd.read_csv(path_school_sup)

    df_colleges["name_clean"] = df_colleges["name"].str.lower().str.strip()
    if "standard_college" in df_programs.columns:
        df_programs["school_name_clean"] = df_programs["standard_college"].str.lower().str.strip()
    else:
        df_programs["school_name_clean"] = df_programs["name"].str.lower().str.strip()
    df_school_sup["school_name_clean"] = df_school_sup["INSTNM"].str.lower().str.strip()

    coverage_dict = {}
    zero_match_programs = []

    all_programs = sorted(df_programs["program"].unique().tolist())

    for prog_name in all_programs:
        df_prog_filtered = df_programs[df_programs["program"] == prog_name]
        merged = pd.merge(
            df_prog_filtered,
            df_colleges,
            left_on="school_name_clean",
            right_on="name_clean",
            how="inner"
        )
        merged = pd.merge(
            merged,
            df_school_sup,
            on="school_name_clean",
            how="inner"
        )
        count_schools = len(merged)
        coverage_dict[prog_name] = count_schools
        if count_schools == 0:
            zero_match_programs.append(prog_name)

    return {
        "total_programs": len(all_programs),
        "coverage": coverage_dict,
        "zero_match_programs": zero_match_programs,
        "num_zero_match": len(zero_match_programs)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0"
    }