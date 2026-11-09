"""
metrics.py
Module responsible for computing analytical KPIs, academic performance metrics,
grade progression statistics, at-risk student identification, and correlation matrices.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level Key Performance Indicators (KPIs) for the current student population.
    
    Returns:
        dict: Summary statistics including counts, averages, pass rates, and risk indices.
    """
    if df.empty:
        return {
            "total_students": 0,
            "avg_final_grade": 0.0,
            "avg_final_percentage": 0.0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "avg_g1": 0.0,
            "avg_g2": 0.0,
            "avg_absences": 0.0,
            "highest_grade": 0.0,
            "lowest_grade": 0.0,
            "at_risk_count": 0,
            "at_risk_rate": 0.0
        }

    total = len(df)
    avg_g3 = float(df["G3"].mean())
    avg_pct = float(df["final_percentage"].mean())
    
    pass_count = int((df["G3"] >= 10).sum())
    fail_count = total - pass_count
    pass_rate = (pass_count / total) * 100.0 if total > 0 else 0.0
    fail_rate = 100.0 - pass_rate

    avg_g1 = float(df["G1"].mean())
    avg_g2 = float(df["G2"].mean())
    avg_absences = float(df["absences"].mean())
    highest = float(df["G3"].max())
    lowest = float(df["G3"].min())

    # Pre-calculated at-risk count (G3 < 10 or absences > 10)
    at_risk_mask = (df["G3"] < 10) | (df["absences"] > 10) | (df["failures"] > 0)
    at_risk_count = int(at_risk_mask.sum())
    at_risk_rate = (at_risk_count / total) * 100.0 if total > 0 else 0.0

    return {
        "total_students": total,
        "avg_final_grade": round(avg_g3, 2),
        "avg_final_percentage": round(avg_pct, 1),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 1),
        "fail_rate": round(fail_rate, 1),
        "avg_g1": round(avg_g1, 2),
        "avg_g2": round(avg_g2, 2),
        "avg_absences": round(avg_absences, 1),
        "highest_grade": round(highest, 1),
        "lowest_grade": round(lowest, 1),
        "at_risk_count": at_risk_count,
        "at_risk_rate": round(at_risk_rate, 1)
    }


def calculate_progression_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes how students progressed from Term 1 (G1) to Term 2 (G2) and Final (G3).
    """
    if df.empty:
        return {
            "improved_count": 0,
            "declined_count": 0,
            "maintained_count": 0,
            "improved_pct": 0.0,
            "declined_pct": 0.0,
            "maintained_pct": 0.0,
            "avg_delta": 0.0,
            "term_averages": {"Term 1 (G1)": 0.0, "Term 2 (G2)": 0.0, "Final (G3)": 0.0}
        }

    total = len(df)
    improved = int((df["progression"] > 0).sum())
    declined = int((df["progression"] < 0).sum())
    maintained = int((df["progression"] == 0).sum())

    return {
        "improved_count": improved,
        "declined_count": declined,
        "maintained_count": maintained,
        "improved_pct": round((improved / total) * 100.0, 1),
        "declined_pct": round((declined / total) * 100.0, 1),
        "maintained_pct": round((maintained / total) * 100.0, 1),
        "avg_delta": round(float(df["progression"].mean()), 2),
        "term_averages": {
            "Term 1 (G1)": round(float(df["G1"].mean()), 2),
            "Term 2 (G2)": round(float(df["G2"].mean()), 2),
            "Final (G3)": round(float(df["G3"].mean()), 2)
        }
    }


def identify_at_risk_students(
    df: pd.DataFrame,
    min_pass_grade: float = 10.0,
    max_allowed_absences: int = 10
) -> pd.DataFrame:
    """
    Flags students needing academic intervention based on transparent, multi-factor criteria.
    
    Risk Factors Considered:
    1. Academic Failure: G3 < min_pass_grade
    2. Continuous Underperformance: G1 < min_pass_grade and G2 < min_pass_grade
    3. Chronic Absenteeism: absences > max_allowed_absences
    4. Academic History: failures > 0 (prior failed courses)
    5. Sharp Decline: G3 - G1 <= -3 (severe drop in performance)
    
    Risk Levels:
    - High Risk: Academic Failure combined with absenteeism or past failures.
    - Moderate Risk: Academic Failure alone, OR high absenteeism with borderline scores.
    - Watchlist: Passing, but showing sharp downward performance trajectory.
    """
    if df.empty:
        return pd.DataFrame()

    records = []

    for _, row in df.iterrows():
        reasons = []
        is_failing_g3 = row["G3"] < min_pass_grade
        is_failing_terms = (row["G1"] < min_pass_grade) and (row["G2"] < min_pass_grade)
        is_chronic_absent = row["absences"] > max_allowed_absences
        has_past_failures = row["failures"] > 0
        has_sharp_decline = (row["G3"] - row["G1"]) <= -3

        if is_failing_g3:
            reasons.append(f"Failing final grade ({row['G3']:.0f}/20)")
        elif is_failing_terms:
            reasons.append(f"Failing preliminary terms (G1={row['G1']:.0f}, G2={row['G2']:.0f})")

        if is_chronic_absent:
            reasons.append(f"High absence count ({row['absences']} absences recorded)")

        if has_past_failures:
            reasons.append(f"Prior class failures ({row['failures']})")

        if has_sharp_decline:
            reasons.append(f"Sharp score drop ({row['G3'] - row['G1']:.0f} pts from Term 1)")

        # Determine risk level
        if reasons:
            if is_failing_g3 and (is_chronic_absent or has_past_failures):
                risk_level = "High Risk"
            elif is_failing_g3 or (is_chronic_absent and row["G3"] <= min_pass_grade + 2):
                risk_level = "Moderate Risk"
            else:
                risk_level = "Watchlist"

            record = {
                "student_id": row["student_id"],
                "risk_level": risk_level,
                "risk_reasons": "; ".join(reasons),
                "G1": row["G1"],
                "G2": row["G2"],
                "G3": row["G3"],
                "final_percentage": row["final_percentage"],
                "absences": row["absences"],
                "failures": row["failures"],
                "studytime_label": row["studytime_label"],
                "school_name": row["school_name"],
                "age": row["age"],
                "sex": row["sex"]
            }
            records.append(record)

    risk_df = pd.DataFrame(records)
    if not risk_df.empty:
        # Sort by severity (High Risk first, then Moderate Risk, then lowest G3)
        risk_priority = {"High Risk": 0, "Moderate Risk": 1, "Watchlist": 2}
        risk_df["priority"] = risk_df["risk_level"].map(risk_priority)
        risk_df = risk_df.sort_values(by=["priority", "G3", "absences"], ascending=[True, True, False])
        risk_df = risk_df.drop(columns=["priority"])

    return risk_df


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Pearson correlation between quantitative academic, behavioral, and personal factors.
    """
    numeric_columns = [
        "age", "studytime", "failures", "famrel", "freetime",
        "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"
    ]
    available_cols = [c for c in numeric_columns if c in df.columns]
    corr = df[available_cols].corr()
    return corr.round(2)


def get_student_profile(df: pd.DataFrame, student_id: str) -> Optional[Dict[str, Any]]:
    """
    Gathers detailed diagnostic information for an individual student.
    
    Includes academic rankings, comparison with class averages, strengths, and warnings.
    """
    if df.empty or "student_id" not in df.columns:
        return None

    student_records = df[df["student_id"] == student_id]
    if student_records.empty:
        return None

    student = student_records.iloc[0]
    total_students = len(df)

    # Class rank based on final grade G3 (descending)
    rank = int(df["G3"].rank(ascending=False, method="min")[df["student_id"] == student_id].values[0])
    percentile = round(((total_students - rank) / total_students) * 100.0, 1)

    # Class averages for comparison
    class_averages = {
        "G1": round(float(df["G1"].mean()), 2),
        "G2": round(float(df["G2"].mean()), 2),
        "G3": round(float(df["G3"].mean()), 2),
        "absences": round(float(df["absences"].mean()), 1),
        "studytime": round(float(df["studytime"].mean()), 2)
    }

    # Strengths and areas of concern
    strengths = []
    concerns = []

    if student["G3"] >= class_averages["G3"]:
        strengths.append(f"Scored {student['G3'] - class_averages['G3']:.1f} pts above class final exam average.")
    else:
        concerns.append(f"Scored {class_averages['G3'] - student['G3']:.1f} pts below class final exam average.")

    if student["absences"] < class_averages["absences"]:
        strengths.append(f"Low absence count ({student['absences']} absences vs. class avg of {class_averages['absences']:.1f}).")
    elif student["absences"] > 10:
        concerns.append(f"Elevated absence count ({student['absences']} absences recorded).")

    if student["studytime"] >= 3:
        strengths.append(f"Dedicated study routine ({student['studytime_label']}).")
    elif student["studytime"] == 1:
        concerns.append("Minimal self-study time (< 2 hours per week).")

    if student["progression"] > 0:
        strengths.append(f"Positive learning trajectory (+{student['progression']} pts between Term 1 and Final).")
    elif student["progression"] < 0:
        concerns.append(f"Performance decline ({student['progression']} pts between Term 1 and Final).")

    if student["failures"] == 0:
        strengths.append("Clean academic track record (0 past course failures).")
    else:
        concerns.append(f"History of {student['failures']} prior course failures.")

    return {
        "record": student,
        "rank": rank,
        "percentile": percentile,
        "class_averages": class_averages,
        "strengths": strengths,
        "concerns": concerns
    }
