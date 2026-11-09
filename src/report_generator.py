"""
report_generator.py
Module providing downloadable CSV export utilities and formatted executive summaries
for the Student Performance & Learning Analytics System.
"""

from datetime import datetime
from typing import Any, Dict
import pandas as pd


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Serializes a DataFrame to UTF-8 encoded CSV bytes for Streamlit download buttons.
    """
    return df.to_csv(index=False).encode("utf-8")


def export_at_risk_report_csv(at_risk_df: pd.DataFrame) -> bytes:
    """
    Produces an export-ready CSV specifically highlighting at-risk students and triggers.
    """
    if at_risk_df.empty:
        empty_df = pd.DataFrame(columns=["student_id", "risk_level", "risk_reasons", "G3", "absences"])
        return empty_df.to_csv(index=False).encode("utf-8")

    columns_to_export = [
        "student_id", "risk_level", "risk_reasons",
        "G1", "G2", "G3", "final_percentage",
        "absences", "failures", "studytime_label",
        "school_name", "age", "sex"
    ]
    available_cols = [c for c in columns_to_export if c in at_risk_df.columns]
    return at_risk_df[available_cols].to_csv(index=False).encode("utf-8")


def generate_executive_summary_text(
    df: pd.DataFrame,
    kpis: Dict[str, Any],
    progression: Dict[str, Any]
) -> str:
    """
    Generates a formal, printable Markdown/Text executive summary of cohort performance.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    summary = f"""========================================================================
STUDENT PERFORMANCE & LEARNING ANALYTICS - COHORT EXECUTIVE REPORT
Generated On: {now}
========================================================================

1. COHORT ENROLLMENT & ACADEMIC KPIs
------------------------------------------------------------------------
- Total Students Analyzed : {kpis['total_students']}
- Class Average (Final G3): {kpis['avg_final_grade']:.2f} / 20 ({kpis['avg_final_percentage']:.1f}%)
- Passing Students        : {kpis['pass_count']} ({kpis['pass_rate']:.1f}%)
- Failing Students        : {kpis['fail_count']} ({kpis['fail_rate']:.1f}%)
- Cohort Highest Score    : {kpis['highest_grade']:.1f} / 20
- Cohort Lowest Score     : {kpis['lowest_grade']:.1f} / 20
- Class Average Absences  : {kpis['avg_absences']:.1f} absences

2. TERM PROGRESSION & TRAJECTORY
------------------------------------------------------------------------
- Term 1 Average (G1)     : {progression['term_averages']['Term 1 (G1)']:.2f} / 20
- Term 2 Average (G2)     : {progression['term_averages']['Term 2 (G2)']:.2f} / 20
- Final Exam Average (G3) : {progression['term_averages']['Final (G3)']:.2f} / 20
- Students with Growth    : {progression['improved_count']} ({progression['improved_pct']:.1f}%)
- Students with Decline   : {progression['declined_count']} ({progression['declined_pct']:.1f}%)
- Students Stable         : {progression['maintained_count']} ({progression['maintained_pct']:.1f}%)
- Net Progression Delta   : {progression['avg_delta']:+.2f} points

3. EARLY WARNING & AT-RISK POPULATION
------------------------------------------------------------------------
- Students Flagged At-Risk: {kpis['at_risk_count']} ({kpis['at_risk_rate']:.1f}% of cohort)
- Primary Trigger Factors : Failing scores (G3 < 10), High Absence Count (>10 absences),
                            and Past Academic Failures.

4. ACTIONABLE EDUCATIONAL RECOMMENDATIONS
------------------------------------------------------------------------
- Attendance Interventions: Institute an automated notification threshold at 8 absences.
- Study Time Guidance     : Encourage students currently studying <2 hrs/week to participate
                            in peer mentoring and structured study tables.
- Remedial Math Support   : Provide targeted tutorial workshops for students with Term 1 (G1)
                            scores below 10 before Term 2 assessments.

========================================================================
Report produced by Student Performance Analyser (Data Science & AI Suite)
========================================================================
"""
    return summary


def generate_student_report_card(student_dict: Dict[str, Any]) -> str:
    """
    Creates a printable individual academic report card for a single student.
    """
    rec = student_dict["record"]
    now = datetime.now().strftime("%Y-%m-%d")

    report = f"""======================================================
           STUDENT ACADEMIC REPORT CARD
======================================================
Report Date : {now}
Student ID  : {rec['student_id']} (Anonymous Generated Identifier)
Institution : {rec['school_name']}
Notice      : Original UCI dataset is strictly anonymized. No personal student names exist.
Age / Sex   : {rec['age']} years / {rec['sex']}
Location    : {rec['address_type']}
------------------------------------------------------
EXAMINATION RESULTS (Scale: 0 to 20):
- Term 1 Assessment (G1)  : {rec['G1']} / 20
- Term 2 Assessment (G2)  : {rec['G2']} / 20
- Final Examination (G3)  : {rec['G3']} / 20 ({rec['final_percentage']:.1f}%)
- Letter Grade Band       : {rec['grade_band']}
- Final Academic Status   : {rec['status']}
- Class Standing          : Rank {student_dict['rank']} (Top {100 - student_dict['percentile']:.1f}% / {student_dict['percentile']}th Percentile)

BEHAVIORAL & LEARNING FACTORS:
- Total Class Absences    : {rec['absences']} ({rec['absence_tier']})
- Weekly Self-Study Time  : {rec['studytime_label']}
- Prior Class Failures    : {rec['failures']}
- Extra School Tutoring   : {rec['schoolsup']}
- Family Academic Support : {rec['famsup']}
- Home Internet Access    : {rec['internet']}

DIAGNOSTIC OBSERVATIONS:
Strengths:
"""
    for s in student_dict["strengths"]:
        report += f"  [+] {s}\n"

    if student_dict["concerns"]:
        report += "Areas for Intervention:\n"
        for c in student_dict["concerns"]:
            report += f"  [-] {c}\n"
    else:
        report += "Areas for Intervention:\n  [+] No urgent flags identified.\n"

    report += "======================================================\n"
    return report
