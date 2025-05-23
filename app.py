from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import json
import math

from services.recommendation_service import (
    recommend_schools as recommend_schools_v1,  # Alias V1 function
    VALID_PROGRAMS, get_school_statistics,
    DF_COLLEGES, DF_PROGRAMS, DF_SCHOOL_SUP, DF_COMPANIES, DF_SCHOOL_IMAGES, 
    SCHOOL_TO_COMPANIES, admissions_data_dict, # V1 globals for V2 init
    load_colleges_data, load_programs_data, load_school_sup_data, 
    load_companies_data_and_build_map, load_admissions_data, load_school_images_data # Ensure all loaders are imported
)
from services.recommendation_service_v2 import recommend_schools_v2, initialize_v2_data, STATE_TO_REGION_MAP, get_admission_statistics_v2
from services.act_sat_conversion import act_to_sat

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

@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup...")
    # Ensure V1 data is loaded first by explicitly calling loaders if not already loaded by import
    if DF_COLLEGES.empty:
        print("V1 DF_COLLEGES not loaded, attempting to load...")
        load_colleges_data()
    if DF_PROGRAMS.empty:
        print("V1 DF_PROGRAMS not loaded, attempting to load...")
        load_programs_data()
    if DF_SCHOOL_SUP.empty:
        print("V1 DF_SCHOOL_SUP not loaded, attempting to load...")
        load_school_sup_data()
    if not SCHOOL_TO_COMPANIES:
        print("V1 SCHOOL_TO_COMPANIES not loaded, attempting to load...")
        load_companies_data_and_build_map()
    if not admissions_data_dict:
        print("V1 admissions_data_dict not loaded, attempting to load...")
        load_admissions_data()
    if DF_SCHOOL_IMAGES.empty:
        print("V1 DF_SCHOOL_IMAGES not loaded, attempting to load...")
        load_school_images_data()

    # DEBUG: Check columns of V1 DataFrames before passing to V2 init
    if not DF_COLLEGES.empty:
        print(f"App Startup: DF_COLLEGES columns from V1 service: {DF_COLLEGES.columns.tolist()}")
    else:
        print("App Startup: DF_COLLEGES from V1 service is EMPTY.")
    
    if not DF_PROGRAMS.empty:
        print(f"App Startup: DF_PROGRAMS columns from V1 service: {DF_PROGRAMS.columns.tolist()}")
    else:
        print("App Startup: DF_PROGRAMS from V1 service is EMPTY.")

    if not DF_COLLEGES.empty and not DF_PROGRAMS.empty:
        # Check for the specific clean name columns before calling init
        if 'name_clean' not in DF_COLLEGES.columns:
            print("CRITICAL STARTUP WARNING: 'name_clean' missing in V1 DF_COLLEGES before V2 init!")
        if 'school_name_clean' not in DF_PROGRAMS.columns:
            print("CRITICAL STARTUP WARNING: 'school_name_clean' missing in V1 DF_PROGRAMS before V2 init!")
            
        initialize_v2_data(
            df_colleges=DF_COLLEGES,
            df_programs=DF_PROGRAMS,
            df_school_sup=DF_SCHOOL_SUP,
            school_to_companies=SCHOOL_TO_COMPANIES,
            admissions_data=admissions_data_dict,
            df_school_images=DF_SCHOOL_IMAGES,
            state_region_map=STATE_TO_REGION_MAP 
        )
    else:
        print("Startup Error: V1 DataFrames (DF_COLLEGES or DF_PROGRAMS) are empty. V2 data initialization failed.")
    print("FastAPI application startup complete.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Models (V1)
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
    value: Any 

class AdmissionYearStats(BaseModel):
    year: int
    metrics: Dict[str, Any]

class Recommendation(BaseModel): 
    School: str
    Recommendation_Tier: str
    Has_Salary_Data: bool
    Median_Earnings_1yr: Optional[float] = None
    Median_Earnings_5yr: Optional[float] = None
    Avg_GPA: Optional[float] = None
    Avg_SAT: Optional[float] = None
    Fortune500_Hirers: List[Dict[str, Any]]
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
    gpa_link: Optional[str] = None
    type_of_institution: Optional[str] = None
    level_of_institution: Optional[str] = None

class RecommendationResponse(BaseModel): 
    recommendations: List[Recommendation]
    timestamp: str
    total_schools: int

# =====================================================
# Models (V2)
# =====================================================
class StudentProfileV2(BaseModel):
    gpa: float = Field(..., example=3.7)
    sat: Optional[int] = Field(None, example=1350)
    act: Optional[float] = Field(None, example=29)

class LocationPreferencesV2(BaseModel):
    states: Optional[List[str]] = Field(None, example=["CA", "NY"])
    region: Optional[str] = Field(None, example="Northeast")

class CostPreferencesV2(BaseModel):
    max_net_price_per_year: Optional[int] = Field(None, example=40000)
    importance: str = Field("medium", example="high")

class PreferencesV2(BaseModel):
    academic_focus: str = Field("match", example="challenge_me")
    location: Optional[LocationPreferencesV2] = None
    cost: Optional[CostPreferencesV2] = None
    school_size: Optional[List[str]] = Field(None, example=["medium", "large"])
    school_type: Optional[str] = Field(None, example="public")
    career_outcomes_importance: str = Field("medium", example="high")
    selectivity_preference: str = Field("any", example="moderate")
    allow_fuzzy_program_match: bool = Field(False)

class RecommendationRequestV2(BaseModel):
    student_profile: StudentProfileV2
    program_query: str = Field(..., example="Computer Science")
    preferences: PreferencesV2
    number_of_recommendations: int = Field(10, gt=0, le=50) # Max 50 as per discussion

class RecommendationV2(BaseModel): 
    School: str 
    Program_Name: str
    V2_Recommendation_Tier: str
    Composite_Score: float
    Why_This_School_Snippet: Optional[str] = None
    
    Recommendation_Tier: Optional[str] = None 
    Has_Salary_Data: Optional[bool] = None
    Median_Earnings_1yr: Optional[float] = None
    Median_Earnings_5yr: Optional[float] = None
    Avg_GPA: Optional[float] = None
    Avg_SAT: Optional[float] = None
    Fortune500_Hirers: List[Dict[str, Any]] = []
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
    gpa_link: Optional[str] = None
    type_of_institution: Optional[str] = None
    level_of_institution: Optional[str] = None
    school_name_clean_key: Optional[str] = None 

    s_academic: Optional[float] = None
    s_program_outcome: Optional[float] = None
    s_affordability: Optional[float] = None
    s_location: Optional[float] = None
    s_selectivity: Optional[float] = None
    s_environment: Optional[float] = None
    s_career: Optional[float] = None

class RecommendationResponseV2(BaseModel):
    recommendations: List[RecommendationV2]
    query_details: RecommendationRequestV2 
    timestamp: str
    total_schools_considered: int 
    total_recommendations_returned: int

# Models for V1 school statistics endpoints (ensure they remain)
class SchoolProgramSalaryInfo(BaseModel):
    program_name: str
    median_earnings_1yr: Optional[float] = None
    median_earnings_5yr: Optional[float] = None

class SchoolStatsResponse(BaseModel):
    school_name_display: str
    school_name_standardized: str
    
    average_gpa: Optional[float] = None
    average_sat: Optional[float] = None
    admission_rate: Optional[float] = None
    total_enrollment: Optional[int] = None
    undergraduate_enrollment: Optional[int] = None
    average_net_price: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    white_enrollment_percent: Optional[float] = None
    black_enrollment_percent: Optional[float] = None
    hispanic_enrollment_percent: Optional[float] = None
    asian_enrollment_percent: Optional[float] = None

    avg_program_median_earnings_1yr: Optional[float] = None
    avg_program_median_earnings_5yr: Optional[float] = None
    
    programs_offered_count: int
    program_salary_details: List[SchoolProgramSalaryInfo]

    admission_statistics: Optional[List[AdmissionYearStats]] = None
    fortune_500_hirers: List[Dict[str, Any]]
    
    image_cdn_url: Optional[str] = None
    image_thumbnail_url: Optional[str] = None
    
    data_sources_used: List[str]
    query_timestamp: str

# New Granular Pydantic Models
class SchoolSalaryStatsResponse(BaseModel):
    school_name_display: str
    school_name_standardized: str
    avg_program_median_earnings_1yr: Optional[float] = None
    avg_program_median_earnings_5yr: Optional[float] = None
    query_timestamp: str

class SchoolAcademicStatsResponse(BaseModel):
    school_name_display: str
    school_name_standardized: str
    average_gpa: Optional[float] = None
    average_sat: Optional[float] = None
    admission_rate: Optional[float] = None
    image_cdn_url: Optional[str] = None
    image_thumbnail_url: Optional[str] = None
    query_timestamp: str

class SchoolDemographicStatsResponse(BaseModel):
    school_name_display: str
    school_name_standardized: str
    total_enrollment: Optional[int] = None
    undergraduate_enrollment: Optional[int] = None
    white_enrollment_percent: Optional[float] = None
    black_enrollment_percent: Optional[float] = None
    hispanic_enrollment_percent: Optional[float] = None
    asian_enrollment_percent: Optional[float] = None
    query_timestamp: str

class SchoolAdmissionTrendsResponse(BaseModel):
    school_name_display: str
    school_name_standardized: str
    admission_statistics: Optional[List[AdmissionYearStats]] = None
    query_timestamp: str

class SchoolHirerStatsResponse(BaseModel):
    school_name_display: str
    school_name_standardized: str
    fortune_500_hirers: List[Dict[str, Any]]
    query_timestamp: str

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

@app.post("/recommendations") # V1 Endpoint
async def get_recommendations(request: RecommendationRequest):
    """
    Generates a list of recommended schools based on the user's GPA, SAT, program, etc.
    Also attaches the admission trends history to each recommendation.
    """
    try:
        df_recommendations = recommend_schools_v1( # Call aliased V1 function
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
        
        if recommendations_list:
            print("\nDebug V1: First recommendation structure:")
            print(json.dumps(recommendations_list[0], indent=2))
        
        return RecommendationResponse(
            recommendations=recommendations_list,
            timestamp=datetime.now().isoformat(),
            total_schools=len(recommendations_list)
        )

    except HTTPException as http_exc:
        raise http_exc
        
    except Exception as e:
        import traceback
        print(f"\nError details V1: {str(e)}")
        print(f"Error type {type(e)}: {str(e)}") # Added type for clarity
        print(f"Traceback V1:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating V1 recommendations: {str(e)}\n{traceback.format_exc()}"
        )

# --- V2 Endpoint ---
@app.post("/v2/recommendations", response_model=RecommendationResponseV2)
async def get_recommendations_v2(request: RecommendationRequestV2):
    """
    Generates V2 university recommendations based on multi-factor scoring.
    """
    try:
        effective_sat = request.student_profile.sat
        if not effective_sat and request.student_profile.act and request.student_profile.act > 0:
            converted_sat = act_to_sat(request.student_profile.act)
            if converted_sat is not None:
                effective_sat = converted_sat
            else:
                effective_sat = None 
        
        student_profile_dict = request.student_profile.model_dump()
        student_profile_dict['effective_sat'] = effective_sat

        df_recs_v2, total_considered = recommend_schools_v2(
            student_profile=student_profile_dict,
            program_query=request.program_query,
            preferences=request.preferences.model_dump(),
            number_of_recommendations=request.number_of_recommendations
        )

        if df_recs_v2 is None or df_recs_v2.empty:
            return RecommendationResponseV2(
                recommendations=[],
                query_details=request,
                timestamp=datetime.now().isoformat(),
                total_schools_considered=total_considered if 'total_considered' in locals() else 0, 
                total_recommendations_returned=0
            )
        
        recs_list_of_dicts = df_recs_v2.to_dict(orient="records")
        cleaned_recs_list = clean_for_json(recs_list_of_dicts) 

        final_recs_v2_models: List[RecommendationV2] = []
        for rec_data in cleaned_recs_list:
            # Ensure Admission_Statistics are correctly fetched for V2
            # The V2 service's returned DataFrame should ideally already have this from `get_admission_statistics_v2`
            admission_stats_data = rec_data.get("Admission_Statistics", []) 
            
            # Ensure Fortune500_Hirers is correctly fetched/formatted
            fortune_hirers_data = rec_data.get("Fortune500_Hirers", [])

            v2_model_data = {
                "School": rec_data.get("name"), 
                "Program_Name": rec_data.get("program_name"),
                "V2_Recommendation_Tier": rec_data.get("V2_Recommendation_Tier", "N/A"),
                "Composite_Score": rec_data.get("composite_score", 0.0),
                "Why_This_School_Snippet": rec_data.get("Why_This_School_Snippet"),
                
                "Recommendation_Tier": rec_data.get("V2_Recommendation_Tier"), # Mapping V2 tier to V1 for now
                "Has_Salary_Data": bool(pd.notna(rec_data.get("earn_mdn_1yr")) or pd.notna(rec_data.get("earn_mdn_5yr"))),
                "Median_Earnings_1yr": rec_data.get("earn_mdn_1yr"),
                "Median_Earnings_5yr": rec_data.get("earn_mdn_5yr"),
                "Avg_GPA": rec_data.get("average_gpa"),
                "Avg_SAT": rec_data.get("average_sat_composite"),
                "Fortune500_Hirers": fortune_hirers_data,
                "Total_Enrollment": rec_data.get("number_of_students"),
                "Admission_Rate": rec_data.get("acceptance_rate"), 
                "Avg_Net_Price": rec_data.get("average_net_price"),
                "Latitude": rec_data.get("LATITUDE"),
                "Longitude": rec_data.get("LONGITUDE"),
                "Admission_Statistics": admission_stats_data,
                "Undergraduate_Enrollment": rec_data.get("UGDS"),
                "White_Enrollment_Percent": rec_data.get("UGDS_WHITE"),
                "Black_Enrollment_Percent": rec_data.get("UGDS_BLACK"),
                "Hispanic_Enrollment_Percent": rec_data.get("UGDS_HISP"),
                "Asian_Enrollment_Percent": rec_data.get("UGDS_ASIAN"),
                "gpa_link": rec_data.get("gpa_link"),
                "type_of_institution": rec_data.get("type_of_institution"),
                "level_of_institution": rec_data.get("level_of_institution"),
                "school_name_clean_key": rec_data.get("school_name_clean"),
                "s_academic": rec_data.get("s_academic"),
                "s_program_outcome": rec_data.get("s_program_outcome"),
                "s_affordability": rec_data.get("s_affordability"),
                "s_location": rec_data.get("s_location"),
                "s_selectivity": rec_data.get("s_selectivity"),
                "s_environment": rec_data.get("s_environment"),
                "s_career": rec_data.get("s_career"),
            }
            try:
                final_recs_v2_models.append(RecommendationV2(**v2_model_data))
            except Exception as pydantic_error:
                # Use rec_data for name if cleaned_rec_data might not be defined due to error in clean_for_json
                school_name_for_error = rec_data.get('name', 'Unknown School') 
                if isinstance(cleaned_rec_data, dict): # Check if cleaned_rec_data is a dict before trying .get()
                    school_name_for_error = cleaned_rec_data.get('name', school_name_for_error)
                print(f"Pydantic Error creating RecommendationV2 for {school_name_for_error}: {pydantic_error}")
                # Optionally, log more details about rec_data for debugging
                # print(f"Problematic rec_data: {rec_data}")
                # print(f"Problematic v2_model_data: {v2_model_data}")

        return RecommendationResponseV2(
            recommendations=final_recs_v2_models,
            query_details=request,
            timestamp=datetime.now().isoformat(),
            total_schools_considered=total_considered, 
            total_recommendations_returned=len(final_recs_v2_models)
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        import traceback
        print(f"\nError in /v2/recommendations: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error in V2 recommendations: {str(e)}")

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

@app.get("/school/{school_name_query}", response_model=SchoolStatsResponse)
async def get_school_stats(school_name_query: str):
    """
    Provides detailed statistics for a requested school, including average salaries 
    across its programs, demographic data, admission trends, and general info.
    """
    try:
        school_data = get_school_statistics(school_name_query)
        if not school_data:
            raise HTTPException(
                status_code=404,
                detail=f"School matching '{school_name_query}' not found or data is insufficient."
            )
        
        # Ensure data is JSON serializable (especially NaN -> None)
        cleaned_school_data = clean_for_json(school_data)
        cleaned_school_data["query_timestamp"] = datetime.now().isoformat()

        return SchoolStatsResponse(**cleaned_school_data)
        
    except HTTPException as http_exc:
        raise http_exc # Re-raise known HTTP exceptions
    except Exception as e:
        import traceback
        # Consider logging the error here instead of printing if you have a logger setup
        print(f"\nError details in /school endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching school statistics: {str(e)}"
        )

@app.get("/school/{school_name_query}/salary", response_model=SchoolSalaryStatsResponse)
async def get_school_salary_stats(school_name_query: str):
    school_data = get_school_statistics(school_name_query)
    if not school_data:
        raise HTTPException(status_code=404, detail=f"School '{school_name_query}' not found.")
    cleaned_data = clean_for_json(school_data)
    return SchoolSalaryStatsResponse(
        school_name_display=cleaned_data["school_name_display"],
        school_name_standardized=cleaned_data["school_name_standardized"],
        avg_program_median_earnings_1yr=cleaned_data.get("avg_program_median_earnings_1yr"),
        avg_program_median_earnings_5yr=cleaned_data.get("avg_program_median_earnings_5yr"),
        query_timestamp=datetime.now().isoformat()
    )

@app.get("/school/{school_name_query}/academics", response_model=SchoolAcademicStatsResponse)
async def get_school_academic_stats(school_name_query: str):
    school_data = get_school_statistics(school_name_query)
    if not school_data:
        raise HTTPException(status_code=404, detail=f"School '{school_name_query}' not found.")
    cleaned_data = clean_for_json(school_data)
    return SchoolAcademicStatsResponse(
        school_name_display=cleaned_data["school_name_display"],
        school_name_standardized=cleaned_data["school_name_standardized"],
        average_gpa=cleaned_data.get("average_gpa"),
        average_sat=cleaned_data.get("average_sat"),
        admission_rate=cleaned_data.get("admission_rate"),
        image_cdn_url=cleaned_data.get("image_cdn_url"),
        image_thumbnail_url=cleaned_data.get("image_thumbnail_url"),
        query_timestamp=datetime.now().isoformat()
    )

@app.get("/school/{school_name_query}/demographics", response_model=SchoolDemographicStatsResponse)
async def get_school_demographic_stats(school_name_query: str):
    school_data = get_school_statistics(school_name_query)
    if not school_data:
        raise HTTPException(status_code=404, detail=f"School '{school_name_query}' not found.")
    cleaned_data = clean_for_json(school_data)
    return SchoolDemographicStatsResponse(
        school_name_display=cleaned_data["school_name_display"],
        school_name_standardized=cleaned_data["school_name_standardized"],
        total_enrollment=cleaned_data.get("total_enrollment"),
        undergraduate_enrollment=cleaned_data.get("undergraduate_enrollment"),
        white_enrollment_percent=cleaned_data.get("white_enrollment_percent"),
        black_enrollment_percent=cleaned_data.get("black_enrollment_percent"),
        hispanic_enrollment_percent=cleaned_data.get("hispanic_enrollment_percent"),
        asian_enrollment_percent=cleaned_data.get("asian_enrollment_percent"),
        query_timestamp=datetime.now().isoformat()
    )

@app.get("/school/{school_name_query}/admission_trends", response_model=SchoolAdmissionTrendsResponse)
async def get_school_admission_trends(school_name_query: str):
    school_data = get_school_statistics(school_name_query)
    if not school_data:
        raise HTTPException(status_code=404, detail=f"School '{school_name_query}' not found.")
    cleaned_data = clean_for_json(school_data)
    return SchoolAdmissionTrendsResponse(
        school_name_display=cleaned_data["school_name_display"],
        school_name_standardized=cleaned_data["school_name_standardized"],
        admission_statistics=cleaned_data.get("admission_statistics", []),
        query_timestamp=datetime.now().isoformat()
    )

@app.get("/school/{school_name_query}/hirers", response_model=SchoolHirerStatsResponse)
async def get_school_hirer_stats(school_name_query: str):
    school_data = get_school_statistics(school_name_query)
    if not school_data:
        raise HTTPException(status_code=404, detail=f"School '{school_name_query}' not found.")
    cleaned_data = clean_for_json(school_data)
    return SchoolHirerStatsResponse(
        school_name_display=cleaned_data["school_name_display"],
        school_name_standardized=cleaned_data["school_name_standardized"],
        fortune_500_hirers=cleaned_data.get("fortune_500_hirers", []),
        query_timestamp=datetime.now().isoformat()
    )