"""
data_loader.py
Module responsible for data ingestion, schema validation, feature enrichment,
and dataset filtering for the Student Performance & Learning Analytics System.
Supports file path loading, custom CSV uploads, and robust sample fallback generation.
"""

import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Default expected file path
DEFAULT_DATASET_PATH = os.path.join("data", "student-mat.csv")

# Core expected columns in UCI student dataset
EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
]


def load_student_data(source: Optional[Union[str, io.BytesIO, Any]] = None) -> pd.DataFrame:
    """
    Loads, cleans, validates and enriches student dataset from a file path or uploaded file.
    Falls back gracefully to a clearly labeled synthetic sample dataset if file is absent.
    
    Parameters:
        source: File path (str), file-like object (UploadedFile / BytesIO), or None.
        
    Returns:
        pd.DataFrame: Cleaned, validated, and enriched DataFrame.
    """
    is_sample = False

    if source is not None and not isinstance(source, str):
        # Source is an uploaded file buffer
        try:
            # First try semicolon delimiter (UCI standard)
            df = pd.read_csv(source, sep=";")
            if len(df.columns) < 5:
                # Fall back to comma
                source.seek(0)
                df = pd.read_csv(source, sep=",")
        except Exception as exc:
            raise ValueError(f"Failed to parse uploaded CSV: {str(exc)}") from exc
    else:
        # Source is a file path or default path
        filepath = source if isinstance(source, str) else DEFAULT_DATASET_PATH
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, sep=";")
                if len(df.columns) < 5:
                    df = pd.read_csv(filepath, sep=",")
            except Exception as exc:
                raise ValueError(f"Failed to parse dataset at '{filepath}': {str(exc)}") from exc
        else:
            # Generate clearly labeled synthetic sample dataset
            df = generate_sample_dataset()
            is_sample = True

    # Clean & Preprocess Data
    df = clean_and_preprocess_dataset(df)

    # Perform Feature Enrichment
    enriched_df = enrich_student_features(df)
    enriched_df["is_sample_data"] = is_sample

    return enriched_df


def clean_and_preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw tabular records, validates types, handles missing values, and ensures schema integrity.
    """
    df = df.copy()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure critical grade and numeric columns exist or provide defaults
    numeric_cols = ["age", "studytime", "failures", "absences", "G1", "G2", "G3"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        else:
            df[col] = 0

    # Clean text columns
    text_cols = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "internet"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace('"', "")
        else:
            df[col] = "Unknown"

    return df


def generate_sample_dataset(n_records: int = 100) -> pd.DataFrame:
    """
    Generates a realistic synthetic sample dataset clearly labeled as demo data.
    Ensures zero downtime if student-mat.csv is temporarily missing.
    """
    np.random.seed(42)

    schools = np.random.choice(["GP", "MS"], size=n_records, p=[0.75, 0.25])
    sexes = np.random.choice(["F", "M"], size=n_records, p=[0.52, 0.48])
    ages = np.random.randint(15, 20, size=n_records)
    addresses = np.random.choice(["U", "R"], size=n_records, p=[0.78, 0.22])
    studytimes = np.random.choice([1, 2, 3, 4], size=n_records, p=[0.25, 0.45, 0.20, 0.10])
    failures = np.random.choice([0, 1, 2, 3], size=n_records, p=[0.75, 0.15, 0.07, 0.03])
    absences = np.random.poisson(lam=5, size=n_records).clip(0, 45)

    # Correlated academic performance grades
    g1 = (10 + studytimes * 1.5 - failures * 2.0 + np.random.normal(0, 2.5, size=n_records)).clip(0, 20).round()
    g2 = (g1 * 0.8 + np.random.normal(2, 1.8, size=n_records)).clip(0, 20).round()
    g3 = (g2 * 0.85 + np.random.normal(1.5, 2.0, size=n_records)).clip(0, 20).round()

    data = {
        "school": schools,
        "sex": sexes,
        "age": ages,
        "address": addresses,
        "famsize": np.random.choice(["GT3", "LE3"], size=n_records),
        "Pstatus": np.random.choice(["T", "A"], size=n_records, p=[0.85, 0.15]),
        "Medu": np.random.randint(0, 5, size=n_records),
        "Fedu": np.random.randint(0, 5, size=n_records),
        "Mjob": np.random.choice(["other", "services", "at_home", "teacher", "health"], size=n_records),
        "Fjob": np.random.choice(["other", "services", "at_home", "teacher", "health"], size=n_records),
        "reason": np.random.choice(["course", "home", "reputation", "other"], size=n_records),
        "guardian": np.random.choice(["mother", "father", "other"], size=n_records),
        "traveltime": np.random.choice([1, 2, 3, 4], size=n_records),
        "studytime": studytimes,
        "failures": failures,
        "schoolsup": np.random.choice(["yes", "no"], size=n_records, p=[0.15, 0.85]),
        "famsup": np.random.choice(["yes", "no"], size=n_records, p=[0.60, 0.40]),
        "paid": np.random.choice(["yes", "no"], size=n_records, p=[0.45, 0.55]),
        "activities": np.random.choice(["yes", "no"], size=n_records),
        "nursery": np.random.choice(["yes", "no"], size=n_records, p=[0.80, 0.20]),
        "higher": np.random.choice(["yes", "no"], size=n_records, p=[0.90, 0.10]),
        "internet": np.random.choice(["yes", "no"], size=n_records, p=[0.85, 0.15]),
        "romantic": np.random.choice(["yes", "no"], size=n_records),
        "famrel": np.random.randint(1, 6, size=n_records),
        "freetime": np.random.randint(1, 6, size=n_records),
        "goout": np.random.randint(1, 6, size=n_records),
        "Dalc": np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.65, 0.20, 0.08, 0.04, 0.03]),
        "Walc": np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.40, 0.30, 0.15, 0.10, 0.05]),
        "health": np.random.randint(1, 6, size=n_records),
        "absences": absences,
        "G1": g1.astype(int),
        "G2": g2.astype(int),
        "G3": g3.astype(int)
    }
    return pd.DataFrame(data)


def enrich_student_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds analytical features to the dataset without modifying raw variables.
    """
    df = df.copy()

    # 1. Generated Anonymous Student Identifier
    df["student_id"] = [f"STU-{i + 1:04d}" for i in range(len(df))]

    # 2. Academic Passing Status (Passing mark is 10 out of 20)
    df["passed"] = df["G3"] >= 10
    df["status"] = df["passed"].map({True: "Passed", False: "Failed"})

    # 3. Percentage conversions
    df["G1_percentage"] = (df["G1"] / 20.0) * 100.0
    df["G2_percentage"] = (df["G2"] / 20.0) * 100.0
    df["final_percentage"] = (df["G3"] / 20.0) * 100.0

    # 4. Standard Academic Grade Bands
    def assign_grade(score: float) -> str:
        if score >= 16:
            return "A (16-20: Excellent)"
        elif score >= 14:
            return "B (14-15: Very Good)"
        elif score >= 12:
            return "C (12-13: Good)"
        elif score >= 10:
            return "D (10-11: Sufficient)"
        else:
            return "F (0-9: Fail)"

    df["grade_band"] = df["G3"].apply(assign_grade)
    df["grade_letter"] = df["G3"].apply(
        lambda s: "A" if s >= 16 else ("B" if s >= 14 else ("C" if s >= 12 else ("D" if s >= 10 else "F")))
    )

    # 5. Term Progression (G3 - G1)
    df["progression"] = df["G3"] - df["G1"]
    df["progression_type"] = df["progression"].apply(
        lambda x: "Improved" if x > 0 else ("Declined" if x < 0 else "Maintained")
    )

    # 6. Human-readable label mappings
    studytime_map = {
        1: "< 2 hours",
        2: "2 - 5 hours",
        3: "5 - 10 hours",
        4: "> 10 hours"
    }
    df["studytime_label"] = df["studytime"].map(studytime_map).fillna("Unknown")

    school_map = {
        "GP": "GP (Gabriel Pereira Secondary School)",
        "MS": "MS (Mousinho da Silveira Secondary School)"
    }
    df["school_name"] = df["school"].map(school_map).fillna(df["school"])

    address_map = {
        "U": "Urban",
        "R": "Rural"
    }
    df["address_type"] = df["address"].map(address_map).fillna(df["address"])

    # 7. Absence Tiers
    def categorize_absences(absences: int) -> str:
        if absences <= 4:
            return "Low (0-4 absences)"
        elif absences <= 9:
            return "Moderate (5-9 absences)"
        elif absences <= 19:
            return "High (10-19 absences)"
        else:
            return "Severe (20+ absences)"

    df["absence_tier"] = df["absences"].apply(categorize_absences)

    return df


def filter_data(
    df: pd.DataFrame,
    school: str = "All",
    sex: str = "All",
    age_range: Tuple[int, int] = (15, 22),
    studytime_filter: Optional[List[str]] = None,
    internet_filter: str = "All",
    status_filter: str = "All"
) -> pd.DataFrame:
    """
    Applies multi-criteria filters to the student DataFrame.
    """
    filtered = df.copy()

    # School filter (matches full institution name or code)
    if school != "All":
        filtered = filtered[(filtered["school_name"] == school) | (filtered["school"] == school)]

    # Sex filter
    if sex != "All":
        filtered = filtered[filtered["sex"] == sex]

    # Age filter
    min_age, max_age = age_range
    filtered = filtered[(filtered["age"] >= min_age) & (filtered["age"] <= max_age)]

    # Study time filter
    if studytime_filter and len(studytime_filter) > 0 and "All" not in studytime_filter:
        filtered = filtered[filtered["studytime_label"].isin(studytime_filter)]

    # Internet access filter
    if internet_filter != "All":
        filtered = filtered[filtered["internet"] == internet_filter]

    # Status filter
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]

    return filtered


def get_column_metadata() -> Dict[str, Dict[str, str]]:
    """
    Returns a comprehensive dictionary describing all variables in the dataset.
    """
    return {
        "school": {"description": "Enrolled public secondary school institution in Portugal", "type": "Categorical", "values": "GP (Gabriel Pereira Secondary School), MS (Mousinho da Silveira Secondary School)"},
        "sex": {"description": "Student's biological sex", "type": "Categorical", "values": "F (Female), M (Male)"},
        "age": {"description": "Student's age in years", "type": "Numeric", "values": "15 to 22"},
        "address": {"description": "Home address area", "type": "Categorical", "values": "U (Urban), R (Rural)"},
        "famsize": {"description": "Family size", "type": "Categorical", "values": "LE3 (<=3), GT3 (>3)"},
        "Pstatus": {"description": "Parent cohabitation status", "type": "Categorical", "values": "T (Together), A (Apart)"},
        "Medu": {"description": "Mother's education level", "type": "Ordinal (0-4)", "values": "0: None to 4: Higher education"},
        "Fedu": {"description": "Father's education level", "type": "Ordinal (0-4)", "values": "0: None to 4: Higher education"},
        "Mjob": {"description": "Mother's occupation", "type": "Categorical", "values": "teacher, health, services, at_home, other"},
        "Fjob": {"description": "Father's occupation", "type": "Categorical", "values": "teacher, health, services, at_home, other"},
        "reason": {"description": "Reason for choosing school", "type": "Categorical", "values": "home, reputation, course, other"},
        "guardian": {"description": "Student's primary guardian", "type": "Categorical", "values": "mother, father, other"},
        "traveltime": {"description": "Home-to-school travel time", "type": "Ordinal (1-4)", "values": "1: <15m, 2: 15-30m, 3: 30-60m, 4: >1h"},
        "studytime": {"description": "Weekly study time", "type": "Ordinal (1-4)", "values": "1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h"},
        "failures": {"description": "Past class failures", "type": "Numeric", "values": "0 to 4"},
        "schoolsup": {"description": "Extra educational support", "type": "Binary", "values": "yes, no"},
        "famsup": {"description": "Family educational support", "type": "Binary", "values": "yes, no"},
        "paid": {"description": "Extra paid classes for math", "type": "Binary", "values": "yes, no"},
        "activities": {"description": "Extra-curricular activities", "type": "Binary", "values": "yes, no"},
        "nursery": {"description": "Attended nursery school", "type": "Binary", "values": "yes, no"},
        "higher": {"description": "Aims for higher education", "type": "Binary", "values": "yes, no"},
        "internet": {"description": "Internet access at home", "type": "Binary", "values": "yes, no"},
        "romantic": {"description": "In a romantic relationship", "type": "Binary", "values": "yes, no"},
        "famrel": {"description": "Quality of family relationships", "type": "Scale (1-5)", "values": "1: Very Bad to 5: Excellent"},
        "freetime": {"description": "Free time after school", "type": "Scale (1-5)", "values": "1: Very Low to 5: Very High"},
        "goout": {"description": "Going out with friends", "type": "Scale (1-5)", "values": "1: Very Low to 5: Very High"},
        "Dalc": {"description": "Workday alcohol consumption", "type": "Scale (1-5)", "values": "1: Very Low to 5: Very High"},
        "Walc": {"description": "Weekend alcohol consumption", "type": "Scale (1-5)", "values": "1: Very Low to 5: Very High"},
        "health": {"description": "Current health status", "type": "Scale (1-5)", "values": "1: Very Bad to 5: Very Good"},
        "absences": {"description": "Number of school absences", "type": "Numeric", "values": "0 to 93"},
        "G1": {"description": "First term assessment score", "type": "Score (0-20)", "values": "0 to 20"},
        "G2": {"description": "Second term assessment score", "type": "Score (0-20)", "values": "0 to 20"},
        "G3": {"description": "Final examination grade (Target)", "type": "Score (0-20)", "values": "0 to 20"}
    }
