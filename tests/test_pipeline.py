"""
test_pipeline.py
Exhaustive Quality Assurance Test Suite for the Student Performance & Learning Analytics System.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root directory to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_student_data, filter_data, get_column_metadata
from src.metrics import (
    calculate_kpis,
    calculate_progression_metrics,
    identify_at_risk_students,
    compute_correlation_matrix,
    get_student_profile,
)
from src.visualizations import (
    plot_absences_vs_grades,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_grade_breakdown_donut,
    plot_grade_distribution,
    plot_student_vs_class_bar,
    plot_study_time_impact,
    plot_term_progression,
)
from src.ml_predictor import (
    train_performance_models,
    predict_single_student,
)
from src.report_generator import (
    export_at_risk_report_csv,
    export_dataframe_to_csv,
    generate_executive_summary_text,
    generate_student_report_card,
)


def test_kpi_mathematical_accuracy():
    """Verify that every KPI is calculated accurately from the actual dataset."""
    df = load_student_data()
    kpis = calculate_kpis(df)

    # 1. Total count
    assert kpis["total_students"] == len(df) == 395

    # 2. Score averages
    assert np.isclose(kpis["avg_final_grade"], df["G3"].mean(), atol=0.01)
    assert np.isclose(kpis["avg_g1"], df["G1"].mean(), atol=0.01)
    assert np.isclose(kpis["avg_g2"], df["G2"].mean(), atol=0.01)
    assert np.isclose(kpis["avg_absences"], df["absences"].mean(), atol=0.01)

    # 3. Pass/Fail counts and rates
    expected_pass = int((df["G3"] >= 10).sum())
    expected_fail = len(df) - expected_pass
    assert kpis["pass_count"] == expected_pass == 265
    assert kpis["fail_count"] == expected_fail == 130
    assert np.isclose(kpis["pass_rate"] + kpis["fail_rate"], 100.0, atol=0.1)

    # 4. Bounds
    assert kpis["highest_grade"] == float(df["G3"].max()) == 20.0
    assert kpis["lowest_grade"] == float(df["G3"].min()) == 0.0

    print("[PASS] [1/7] KPI Mathematical Accuracy Verified.")


def test_filters_and_state_transitions():
    """Test all interactive filter combinations and edge cases."""
    df = load_student_data()

    # School filter (supports mapped name or raw code)
    df_gp = filter_data(df, school="GP (Gabriel Pereira Secondary School)")
    df_ms = filter_data(df, school="MS")
    assert len(df_gp) + len(df_ms) == len(df)

    # Gender filter
    df_f = filter_data(df, sex="F")
    df_m = filter_data(df, sex="M")
    assert len(df_f) + len(df_m) == len(df)

    # Age filter
    df_young = filter_data(df, age_range=(15, 16))
    assert all((df_young["age"] >= 15) & (df_young["age"] <= 16))

    # Study time filter
    df_study = filter_data(df, studytime_filter=["> 10 hours"])
    assert all(df_study["studytime"] == 4)

    # Academic status filter
    df_pass = filter_data(df, status_filter="Passed")
    df_fail = filter_data(df, status_filter="Failed")
    assert len(df_pass) == 265
    assert len(df_fail) == 130

    # Edge case: Empty filtered set
    df_empty = filter_data(df, school="GP", age_range=(22, 22), status_filter="Failed")
    kpis_empty = calculate_kpis(df_empty)
    assert kpis_empty["total_students"] == len(df_empty)
    assert kpis_empty["pass_rate"] == 0.0

    # Edge case: Single record
    df_single = df.iloc[:1]
    kpis_single = calculate_kpis(df_single)
    assert kpis_single["total_students"] == 1
    assert kpis_single["pass_count"] in (0, 1)

    print("[PASS] [2/7] Cohort Filters & Edge Cases Verified.")


def test_attendance_absence_counts():
    """Verify that attendance analysis works on actual absence counts without invented percentages."""
    df = load_student_data()

    # Verify column existence and integer absence counts
    assert "absences" in df.columns
    assert pd.api.types.is_integer_dtype(df["absences"]) or pd.api.types.is_numeric_dtype(df["absences"])
    assert df["absences"].min() >= 0
    assert df["absences"].max() <= 93

    # Check Absence Tiers
    assert "absence_tier" in df.columns
    for tier in df["absence_tier"].unique():
        assert "absence" in tier.lower()
        assert "%" not in tier  # No invented percentages

    # Verify correlation calculation
    corr = df["absences"].corr(df["G3"])
    assert not np.isnan(corr)
    assert -1.0 <= corr <= 1.0

    print("[PASS] [3/7] Attendance & Absence Counts Verified.")


def test_student_profiler_and_search():
    """Test individual profile retrieval, ranking, percentile, and report card generation."""
    df = load_student_data()

    # 1. Test STU-0001 exact ground truth data from student-mat.csv
    p1 = get_student_profile(df, "STU-0001")
    assert p1 is not None
    r1 = p1["record"]
    assert r1["G1"] == 5 and r1["G2"] == 6 and r1["G3"] == 6
    assert r1["absences"] == 6
    assert r1["failures"] == 0
    assert r1["school"] == "GP"
    assert r1["sex"] == "F" and r1["age"] == 18
    assert r1["status"] == "Failed"
    assert "GP (Gabriel Pereira Secondary School)" in r1["school_name"]

    # 2. Test STU-0002 exact ground truth data from student-mat.csv
    p2 = get_student_profile(df, "STU-0002")
    assert p2 is not None
    r2 = p2["record"]
    assert r2["G1"] == 5 and r2["G2"] == 5 and r2["G3"] == 6
    assert r2["absences"] == 4
    assert r2["failures"] == 0
    assert r2["school"] == "GP"
    assert r2["sex"] == "F" and r2["age"] == 17
    assert r2["status"] == "Failed"
    assert "GP (Gabriel Pereira Secondary School)" in r2["school_name"]

    # 3. Test Invalid Searches and Empty Selections
    p_invalid = get_student_profile(df, "STU-9999")
    assert p_invalid is None, "Non-existent student ID should return None"

    p_empty_id = get_student_profile(df, "")
    assert p_empty_id is None, "Empty student ID should return None"

    p_empty_df = get_student_profile(pd.DataFrame(), "STU-0001")
    assert p_empty_df is None, "Empty dataframe should return None"

    # 4. Test diverse student profiles
    sample_ids = ["STU-0001", "STU-0048", "STU-0112", "STU-0395"]
    for sid in sample_ids:
        profile = get_student_profile(df, sid)
        assert profile is not None
        assert profile["record"]["student_id"] == sid
        assert 1 <= profile["rank"] <= len(df)
        assert 0.0 <= profile["percentile"] <= 100.0
        assert isinstance(profile["strengths"], list)
        assert isinstance(profile["concerns"], list)

        # Verify report card text
        report_card = generate_student_report_card(profile)
        assert sid in report_card
        assert "STUDENT ACADEMIC REPORT CARD" in report_card
        assert "Anonymous Generated Identifier" in report_card

    print("[PASS] [4/7] Student Profiler & Search Engine Verified.")


def test_at_risk_system_and_reporting():
    """Test transparent multi-factor at-risk detection and exportable reports."""
    df = load_student_data()

    # Default threshold test
    at_risk = identify_at_risk_students(df, min_pass_grade=10.0, max_allowed_absences=10)
    assert not at_risk.empty
    assert "risk_level" in at_risk.columns
    assert "risk_reasons" in at_risk.columns
    assert all(at_risk["risk_level"].isin(["High Risk", "Moderate Risk", "Watchlist"]))

    # Strict threshold test (min grade 12, max absences 5)
    at_risk_strict = identify_at_risk_students(df, min_pass_grade=12.0, max_allowed_absences=5)
    assert len(at_risk_strict) >= len(at_risk)

    # Export CSV tests
    csv_bytes = export_at_risk_report_csv(at_risk)
    assert len(csv_bytes) > 0
    df_roundtrip = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    assert len(df_roundtrip) == len(at_risk)

    # Executive summary report
    kpis = calculate_kpis(df)
    prog = calculate_progression_metrics(df)
    summary_text = generate_executive_summary_text(df, kpis, prog)
    assert "COHORT EXECUTIVE REPORT" in summary_text
    assert f"Total Students Analyzed : {len(df)}" in summary_text

    print("[PASS] [5/7] At-Risk Engine & Reporting Center Verified.")


def test_all_visualizations_robustness():
    """Verify that all 7 charts render correctly across various subset conditions."""
    df = load_student_data()
    profile = get_student_profile(df, "STU-0001")

    # 1. Full cohort
    f1 = plot_grade_distribution(df)
    f2 = plot_term_progression(df)
    f3 = plot_absences_vs_grades(df)
    f4 = plot_study_time_impact(df)
    f5 = plot_correlation_heatmap(df)
    f6 = plot_grade_breakdown_donut(df)
    f7 = plot_student_vs_class_bar(profile["record"], profile["class_averages"])
    for f in [f1, f2, f3, f4, f5, f6, f7]:
        assert f is not None

    # 2. Small filtered subset (e.g. 5 students)
    subset_df = df.head(5)
    f_sub1 = plot_grade_distribution(subset_df)
    f_sub2 = plot_term_progression(subset_df)
    f_sub3 = plot_absences_vs_grades(subset_df)
    f_sub4 = plot_study_time_impact(subset_df)
    f_sub5 = plot_correlation_heatmap(subset_df)
    for f in [f_sub1, f_sub2, f_sub3, f_sub4, f_sub5]:
        assert f is not None

    print("[PASS] [6/7] All Visualizations Verified for Robustness.")


def test_ml_predictor():
    """Verify Scikit-learn predictive modeling, evaluation metrics, and what-if simulation."""
    df = load_student_data()

    # 1. Train models on full benchmark cohort
    model_bundle = train_performance_models(df)
    assert model_bundle["trained"] is True

    # Check evaluation metrics
    metrics = model_bundle["metrics"]
    assert metrics["accuracy"] >= 80.0, f"Accuracy expected >= 80%, got {metrics['accuracy']}%"
    assert metrics["f1_score"] >= 0.75, f"F1 score expected >= 0.75, got {metrics['f1_score']}"
    assert metrics["r2_score"] >= 0.50, f"R2 score expected >= 0.50, got {metrics['r2_score']}"
    assert metrics["rmse"] <= 3.5, f"RMSE expected <= 3.5, got {metrics['rmse']}"

    # Check feature importances
    feat_df = model_bundle["feature_importances"]
    assert not feat_df.empty
    assert "feature" in feat_df.columns
    assert "importance" in feat_df.columns
    assert "feature_name" in feat_df.columns
    top_features = list(feat_df["feature"].head(3))
    # G2 and G1 are known strongest academic predictors in UCI dataset
    assert "G2" in top_features or "G1" in top_features

    # Check predictions DataFrame
    preds_df = model_bundle["predictions_df"]
    assert len(preds_df) == len(df)
    assert "predicted_status" in preds_df.columns
    assert "predicted_pass_prob" in preds_df.columns
    assert "predicted_G3" in preds_df.columns

    # 2. Test single student simulator: High achiever
    high_student = {
        "G1": 18, "G2": 18, "studytime": 4, "failures": 0, "absences": 2,
        "Medu": 4, "Fedu": 4, "traveltime": 1, "famrel": 5, "freetime": 3,
        "goout": 2, "health": 5
    }
    pred_high = predict_single_student(model_bundle, high_student)
    assert pred_high["predicted_pass"] is True
    assert pred_high["pass_probability"] >= 80.0
    assert pred_high["predicted_G3_score"] >= 14.0

    # 3. Test single student simulator: High risk / struggling student
    struggling_student = {
        "G1": 4, "G2": 4, "studytime": 1, "failures": 3, "absences": 22,
        "Medu": 1, "Fedu": 1, "traveltime": 3, "famrel": 2, "freetime": 4,
        "goout": 4, "health": 2
    }
    pred_low = predict_single_student(model_bundle, struggling_student)
    assert pred_low["predicted_pass"] is False
    assert pred_low["pass_probability"] <= 35.0
    assert pred_low["predicted_G3_score"] < 10.0

    # 4. Test feature importance chart generation
    fig_feat = plot_feature_importance(feat_df)
    assert fig_feat is not None

    print("[PASS] [7/7] Machine Learning Predictor & Simulator QA Verified.")


if __name__ == "__main__":
    print("\n=======================================================")
    print("RUNNING EXHAUSTIVE QUALITY ASSURANCE SUITE...")
    print("=======================================================\n")
    test_kpi_mathematical_accuracy()
    test_filters_and_state_transitions()
    test_attendance_absence_counts()
    test_student_profiler_and_search()
    test_at_risk_system_and_reporting()
    test_all_visualizations_robustness()
    test_ml_predictor()
    print("\n=======================================================")
    print(">>> 100% QUALITY REVIEW TESTS PASSED SUCCESSFULLY! <<<")
    print("=======================================================\n")
