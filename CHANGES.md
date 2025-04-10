# ACT Score Support for University Recommendation Server

## Overview
This document describes the changes made to implement ACT score support in the university recommendation server. The implementation allows the system to accept ACT scores and convert them to equivalent SAT scores when SAT scores are not available.

## Implementation Details

### 1. ACT to SAT Conversion Module
A new module `act_sat_conversion.py` was created with functions to convert between ACT and SAT scores based on the official concordance tables from ACT.org.

- `act_to_sat(act_score)`: Converts an ACT Composite score to an equivalent SAT Total score
- `sat_to_act(sat_score)`: Converts an SAT Total score to an equivalent ACT Composite score

The conversion tables were derived from the official ACT-SAT concordance data available at:
https://www.act.org/content/act/en/products-and-services/the-act/scores/act-sat-concordance.html

### 2. Recommendation Service Modifications
The `recommendation_service.py` file was modified to:

- Import the ACT to SAT conversion functions
- Add logic to convert ACT scores to SAT scores when SAT is not provided
- Use the converted SAT score in the recommendation algorithm
- Add robust error handling for missing data columns

### 3. Error Handling Improvements
Several improvements were made to handle potential data quality issues:

- Added checks for missing columns in the companies data file
- Added checks for missing LOCALE data in the composite score calculation
- Added checks for missing columns in various preference calculations
- Added type checking for fortune_500_companies data

## Testing
The implementation was tested with both SAT and ACT scores to verify that:

1. The ACT to SAT conversion functions work correctly
2. The system properly uses the converted score when SAT is not available
3. The code is robust against missing data columns

## Usage
The API endpoints continue to work as before, but now accept an additional `act` parameter. When both SAT and ACT scores are provided, the SAT score takes precedence. When only an ACT score is provided, it is converted to an equivalent SAT score for use in the recommendation algorithm.
