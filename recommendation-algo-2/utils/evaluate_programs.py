import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from tqdm import tqdm
import multiprocessing
import platform

# Add the current directory to Python path to ensure we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the recommend_schools function
try:
    from generate_recommendations2 import recommend_schools
except ImportError as e:
    print(f"Error importing recommend_schools: {e}")
    print("Make sure the file is named 'generate_recommendations2.py' (with underscore, not hyphen)")
    sys.exit(1)

# Global cache for dataframes
_CACHE = {}

def load_data():
    """Load all data files once and cache them"""
    if not _CACHE:
        _CACHE['colleges'] = pd.read_csv("colleges_data_cleaned.csv")
        _CACHE['programs'] = pd.read_csv("programs_cleaned.csv")
        _CACHE['school_sup'] = pd.read_csv("school_sup_data.csv")
        _CACHE['companies'] = pd.read_csv("companies_data_cleaned.csv")
    return _CACHE

def get_unique_programs():
    """Get unique programs from the programs_cleaned.csv file."""
    try:
        programs = pd.read_csv("programs_cleaned.csv")
        return sorted(programs["program"].unique())
    except Exception as e:
        print(f"Error reading programs file: {e}")
        sys.exit(1)

def get_optimal_workers():
    """Get optimal number of workers based on system"""
    if platform.processor() == 'arm':  # M1/M2 Mac
        # M2 has 8 performance cores + 4 efficiency cores
        return 10  # Use 10 workers to maximize performance cores while leaving room for system
    return min(multiprocessing.cpu_count() * 2, 20)

def evaluate_program_batch(args):
    """Evaluate all GPA/SAT combinations for a single program"""
    program, gpa_range, sat_range = args
    
    results = []
    for gpa in gpa_range:
        for sat in sat_range:
            try:
                # Get fresh recommendations for each GPA/SAT combination
                recommendations = recommend_schools(
                    user_gpa=gpa,
                    user_sat=sat,
                    user_program=program,
                    verbose=False
                )
                
                if recommendations is None or recommendations.empty:
                    metrics = {metric: 0 for metric in ['total', 'strong_match', 'good_match', 
                                                      'potential_match', 'consider', 'option', 
                                                      'limited_data']}
                else:
                    # Count schools in each tier based on Recommendation_Tier
                    metrics = {
                        'total': len(recommendations),
                        'strong_match': sum(recommendations['Recommendation_Tier'].str.startswith('Strong Match')),
                        'good_match': sum(recommendations['Recommendation_Tier'].str.startswith('Good Match')),
                        'potential_match': sum(recommendations['Recommendation_Tier'].str.startswith('Potential Match')),
                        'consider': sum(recommendations['Recommendation_Tier'].str.startswith('Consider')),
                        'option': sum(recommendations['Recommendation_Tier'].str.startswith('Option')),
                        'limited_data': sum(recommendations['Recommendation_Tier'].str.contains('Limited Data'))
                    }
                
                results.append((gpa, sat, metrics))
            except Exception as e:
                print(f"Error processing {program} with GPA={gpa}, SAT={sat}: {str(e)}")
                results.append((gpa, sat, None))
    
    return program, results

def main():
    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"program_evaluations_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define ranges
    gpa_range = np.arange(4.0, 0.8, -0.2)  # 4.0 to 1.0
    sat_range = range(800, 1601, 100)  # 800 to 1600
    
    # Get programs and prepare batches
    programs = sorted(pd.read_csv("programs_cleaned.csv")["program"].unique())
    program_batches = [(prog, gpa_range, sat_range) for prog in programs]
    
    # Initialize matrices
    metrics = ['total', 'strong_match', 'good_match', 'potential_match', 
              'consider', 'option', 'limited_data']
    program_matrices = {
        program: {
            metric: np.zeros((len(gpa_range), len(sat_range))) 
            for metric in metrics
        } for program in programs
    }
    
    # Get optimal number of workers for the system
    n_workers = get_optimal_workers()
    
    # Process programs in parallel with optimized batch size
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        # Submit in smaller batches to better utilize M2's efficiency cores
        batch_size = max(1, len(program_batches) // n_workers)
        for i in range(0, len(program_batches), batch_size):
            batch = program_batches[i:i + batch_size]
            for args in batch:
                futures.append(executor.submit(evaluate_program_batch, args))
        
        # Track progress with tqdm
        with tqdm(total=len(programs), desc="Processing programs") as pbar:
            for future in as_completed(futures):
                try:
                    program, results = future.result()
                    
                    # Update matrices for this program
                    for gpa, sat, metrics in results:
                        if metrics is not None:
                            gpa_idx = list(gpa_range).index(gpa)
                            sat_idx = list(sat_range).index(sat)
                            for metric, value in metrics.items():
                                program_matrices[program][metric][gpa_idx, sat_idx] = value
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing a program: {str(e)}")
    
    # Save results using numpy's efficient I/O
    summaries = []
    for program in programs:
        safe_program_name = "".join(c for c in program if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        # Save matrices
        for metric, matrix in program_matrices[program].items():
            df = pd.DataFrame(
                matrix,
                index=[f"GPA {gpa:.1f}" for gpa in gpa_range],
                columns=[f"SAT {sat}" for sat in sat_range]
            )
            matrix_file = os.path.join(output_dir, f"{safe_program_name}_{metric}_matrix.csv")
            df.to_csv(matrix_file)
        
        # Create summary
        summary = {
            "Program": program,
            "Total_Combinations": len(gpa_range) * len(sat_range)
        }
        
        # Add metrics to summary
        for metric, matrix in program_matrices[program].items():
            summary[f"{metric}_max"] = matrix.max()
            summary[f"{metric}_min"] = matrix.min()
            summary[f"{metric}_avg"] = matrix.mean()
            summary[f"{metric}_nonzero"] = np.count_nonzero(matrix)
        
        summaries.append(summary)
    
    # Save overall summary
    pd.DataFrame(summaries).to_csv(os.path.join(output_dir, "program_summary.csv"), index=False)
    
    print(f"Results saved in '{output_dir}'")

if __name__ == "__main__":
    main() 