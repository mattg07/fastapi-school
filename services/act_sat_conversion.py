"""
ACT to SAT Score Conversion Module

This module provides functions to convert between ACT and SAT scores
based on the official concordance tables from ACT.org.
Source: https://www.act.org/content/act/en/products-and-services/the-act/scores/act-sat-concordance.html
"""

def act_to_sat(act_score):
    """
    Convert an ACT Composite score to an equivalent SAT Total score.
    
    Args:
        act_score (float or int): ACT Composite score (1-36)
        
    Returns:
        int: Equivalent SAT Total score (400-1600)
        None: If the input is None or invalid
    """
    if act_score is None:
        return None
        
    # Convert to float first to handle potential string inputs
    try:
        act_score = float(act_score)
    except (ValueError, TypeError):
        return None
    
    # Round to nearest integer for lookup
    act_score = round(act_score)
    
    # Ensure the score is within valid ACT range
    if act_score < 1 or act_score > 36:
        return None
    
    # ACT to SAT conversion table based on official concordance
    conversion_table = {
        36: 1590,  # Using 1590 as the official table shows 36 maps to 1570-1600 range
        35: 1540,
        34: 1500,
        33: 1460,
        32: 1430,
        31: 1400,
        30: 1370,
        29: 1340,
        28: 1310,
        27: 1280,
        26: 1240,
        25: 1210,
        24: 1180,
        23: 1140,
        22: 1110,
        21: 1080,
        20: 1040,
        19: 1010,
        18: 970,
        17: 930,
        16: 890,
        15: 850,
        14: 800,
        13: 760,
        12: 710,
        11: 670,
        10: 630,
        9: 590,
        8: 550,
        7: 510,
        6: 470,
        5: 430,
        4: 400,
        3: 400,  # The table doesn't go below 400 for SAT
        2: 400,
        1: 400
    }
    
    return conversion_table.get(act_score)

def sat_to_act(sat_score):
    """
    Convert an SAT Total score to an equivalent ACT Composite score.
    
    Args:
        sat_score (int): SAT Total score (400-1600)
        
    Returns:
        int: Equivalent ACT Composite score (1-36)
        None: If the input is None or invalid
    """
    if sat_score is None:
        return None
        
    # Convert to int first to handle potential string inputs
    try:
        sat_score = int(sat_score)
    except (ValueError, TypeError):
        return None
    
    # Ensure the score is within valid SAT range
    if sat_score < 400 or sat_score > 1600:
        return None
    
    # SAT score ranges and their corresponding ACT scores
    sat_ranges = [
        (1570, 1600, 36),
        (1530, 1560, 35),
        (1490, 1520, 34),
        (1450, 1480, 33),
        (1420, 1440, 32),
        (1390, 1410, 31),
        (1360, 1380, 30),
        (1330, 1350, 29),
        (1300, 1320, 28),
        (1260, 1290, 27),
        (1230, 1250, 26),
        (1200, 1220, 25),
        (1160, 1190, 24),
        (1130, 1150, 23),
        (1100, 1120, 22),
        (1060, 1090, 21),
        (1030, 1050, 20),
        (990, 1020, 19),
        (960, 980, 18),
        (920, 950, 17),
        (880, 910, 16),
        (830, 870, 15),
        (780, 820, 14),
        (730, 770, 13),
        (690, 720, 12),
        (650, 680, 11),
        (620, 640, 10),
        (590, 610, 9),
        (400, 580, 8)  # Simplifying the lower ranges
    ]
    
    for low, high, act in sat_ranges:
        if low <= sat_score <= high:
            return act
    
    # Default fallback (should not reach here given the input validation)
    return None
