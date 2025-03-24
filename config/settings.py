from pydantic_settings import BaseSettings
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "University Recommendation API"
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # CSV File Paths
    COLLEGES_DATA_PATH: str = str(DATA_DIR / "colleges_data_cleaned.csv")
    PROGRAMS_DATA_PATH: str = str(DATA_DIR / "programs_cleaned.csv")
    SCHOOL_SUP_DATA_PATH: str = str(DATA_DIR / "school_sup_data.csv")
    COMPANIES_DATA_PATH: str = str(DATA_DIR / "companies_data_cleaned.csv")
    
    # API Settings
    CORS_ORIGINS: list = ["http://localhost:3000"]  # Add your Next.js frontend URL
    DEBUG: bool = False
    
    # Cache Settings
    CACHE_TTL: int = 3600  # 1 hour in seconds
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 