"""
app.py
Main Streamlit Application for the Student Performance & Learning Analytics System.
A portfolio-ready data science web app for analyzing academic trends, attendance impacts,
and early-warning student interventions.
"""

import io
import os
import sys
from typing import Optional

# Ensure project root is available in module search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Internal modules
from src.data_loader import (
    DEFAULT_DATASET_PATH,
    filter_data,
    get_column_metadata,
    load_student_data,
)
from src.metrics import (
    calculate_kpis,
    calculate_progression_metrics,
    compute_correlation_matrix,
    get_student_profile,
    identify_at_risk_students,
)
from src.ml_predictor import (
    predict_single_student,
    train_performance_models,
)
from src.report_generator import (
    export_at_risk_report_csv,
    export_dataframe_to_csv,
    generate_executive_summary_text,
    generate_student_report_card,
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


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance & Learning Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
    }
    /* Section Headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data Ingestion with Streamlit Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading academic dataset...")
def get_cached_student_data(file_bytes: Optional[bytes] = None) -> pd.DataFrame:
    """Loads and caches the student dataset from uploaded bytes or default path."""
    if file_bytes is not None:
        return load_student_data(io.BytesIO(file_bytes))
    return load_student_data(DEFAULT_DATASET_PATH)


# -----------------------------------------------------------------------------
# Sidebar: Navigation & Interactive Filters
# -----------------------------------------------------------------------------
st.sidebar.title("🎓 Analytics Hub")
st.sidebar.caption("AI & Data Science Academic Suite")

# Custom CSV Dataset Uploader
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Custom CSV Dataset:",
    type=["csv"],
    help="Upload your own student dataset (semicolon or comma delimited) or leave empty to use the benchmark UCI dataset.",
)

try:
    if uploaded_file is not None:
        raw_df = get_cached_student_data(uploaded_file.getvalue())
        st.sidebar.success(f"Loaded: `{uploaded_file.name}`")
    else:
        raw_df = get_cached_student_data()
except Exception as error:
    st.error(f"❌ Error loading dataset: {str(error)}")
    st.info(
        "Please ensure `data/student-mat.csv` is present in the project directory or upload a valid CSV."
    )
    st.stop()

if "is_sample_data" in raw_df.columns and bool(raw_df["is_sample_data"].iloc[0]):
    st.sidebar.warning("ℹ️ Notice: Using synthetic demonstration dataset because `data/student-mat.csv` was not found.")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Navigation")
nav_selection = st.sidebar.radio(
    label="Go to section:",
    options=[
        "📊 Dashboard Overview",
        "📈 Grade Progression (G1/G2/G3)",
        "🕒 Attendance & Behavior",
        "👤 Student Profile & Search",
        "⚠️ At-Risk Early Warning & ML Forecasting",
        "📥 Export Reports & Data",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Cohort Filters")

# Filter: School Institution (GP and MS are Portuguese public high schools)
school_options = [
    "All",
    "GP (Gabriel Pereira Secondary School)",
    "MS (Mousinho da Silveira Secondary School)",
]
selected_school = st.sidebar.selectbox("School Institution:", school_options, index=0)

# Filter: Sex
sex_options = ["All", "F", "M"]
sex_format = {"All": "All", "F": "Female (F)", "M": "Male (M)"}
selected_sex = st.sidebar.selectbox(
    "Student Sex:", sex_options, format_func=lambda x: sex_format[x]
)

# Filter: Age Range
min_age = int(raw_df["age"].min())
max_age = int(raw_df["age"].max())
selected_age_range = st.sidebar.slider(
    "Age Range:",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age),
    step=1,
)

# Filter: Study Time
study_labels = ["< 2 hours", "2 - 5 hours", "5 - 10 hours", "> 10 hours"]
selected_studytime = st.sidebar.multiselect(
    "Weekly Study Time:",
    options=study_labels,
    default=study_labels,
)

# Filter: Internet Access
internet_options = ["All", "yes", "no"]
internet_labels = {"All": "All", "yes": "Has Internet", "no": "No Internet"}
selected_internet = st.sidebar.selectbox(
    "Home Internet:", internet_options, format_func=lambda x: internet_labels[x]
)

# Filter: Academic Status
status_options = ["All", "Passed", "Failed"]
selected_status = st.sidebar.selectbox("Academic Standing:", status_options)

# Apply Filters
filtered_df = filter_data(
    raw_df,
    school=selected_school,
    sex=selected_sex,
    age_range=selected_age_range,
    studytime_filter=selected_studytime,
    internet_filter=selected_internet,
    status_filter=selected_status,
)

# Display Cohort Counter in Sidebar
st.sidebar.markdown("---")
cohort_pct = (len(filtered_df) / len(raw_df)) * 100 if len(raw_df) else 0.0
st.sidebar.caption("📊 Filter Status")
st.sidebar.info(f"**Showing {len(filtered_df)} of {len(raw_df)} students** ({cohort_pct:.1f}% of cohort)")

if st.sidebar.button("🔄 Reset All Filters", width="stretch"):
    st.rerun()


# -----------------------------------------------------------------------------
# Compute Real-time KPIs for Filtered Population
# -----------------------------------------------------------------------------
kpi_data = calculate_kpis(filtered_df)
progression_data = calculate_progression_metrics(filtered_df)


# -----------------------------------------------------------------------------
# 1. Dashboard Overview
# -----------------------------------------------------------------------------
if nav_selection == "📊 Dashboard Overview":
    st.title("🎓 Student Performance & Learning Analytics")
    st.markdown(
        """
        Welcome to the **Student Performance & Learning Analytics System**. 
        This interactive dashboard visualizes mathematics course outcomes, identifies learning bottlenecks, 
        and monitors student success metrics derived from empirical academic data.
        """
    )

    # Top KPI Metric Cards (Validated against actual dataset, zero unsupported deltas)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Students",
            value=f"{kpi_data['total_students']:,}",
            help="Total number of students in the current filtered cohort.",
        )
    with c2:
        st.metric(
            label="Average Final Grade",
            value=f"{kpi_data['avg_final_grade']:.2f} / 20",
            help=f"Class average score: {kpi_data['avg_final_grade']:.2f} out of 20 ({kpi_data['avg_final_percentage']:.1f}% equivalent).",
        )
    with c3:
        st.metric(
            label="Overall Pass Rate",
            value=f"{kpi_data['pass_rate']:.1f}%",
            help=f"Passing threshold is 10/20. Passed: {kpi_data['pass_count']}, Failed: {kpi_data['fail_count']}.",
        )
    with c4:
        st.metric(
            label="At-Risk Population",
            value=f"{kpi_data['at_risk_count']}",
            help=f"{kpi_data['at_risk_count']} students ({kpi_data['at_risk_rate']:.1f}% of cohort) flagged due to failing marks, high absences (>10), or past failures.",
        )

    st.markdown("---")

    # Visualizations Row
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown(
            '<div class="section-header">📊 Final Grade Distribution (G3)</div>',
            unsafe_allow_html=True,
        )
        if not filtered_df.empty:
            fig_dist = plot_grade_distribution(filtered_df)
            st.pyplot(fig_dist, width="stretch")
            plt.close(fig_dist)
        else:
            st.warning("No student records match the active filter criteria.")

    with col_right:
        st.markdown(
            '<div class="section-header">🍩 Academic Grade Bands</div>',
            unsafe_allow_html=True,
        )
        if not filtered_df.empty:
            fig_donut = plot_grade_breakdown_donut(filtered_df)
            st.pyplot(fig_donut, width="stretch")
            plt.close(fig_donut)
        else:
            st.warning("No student records match the active filter criteria.")

    # Contextual Analytical Summary Box
    st.markdown(
        '<div class="section-header">💡 Key Academic Insights</div>',
        unsafe_allow_html=True,
    )
    with st.expander("🔍 Click to read automated cohort observations", expanded=True):
        st.markdown(
            f"""
            - **Grade Bimodality & Zeros:** The cohort average final grade is **{kpi_data['avg_final_grade']:.2f} / 20**. Notice the cluster of zero scores; in standard educational datasets, these frequently represent students who dropped out or were absent for the final examination.
            - **Passing Ratio:** **{kpi_data['pass_rate']:.1f}%** of students successfully cleared the threshold (>= 10/20).
            - **Absence Index:** Cohort students have an average of **{kpi_data['avg_absences']:.1f} recorded absences**.
            - **Score Range:** Highest score achieved is **{kpi_data['highest_grade']:.0f}/20**, while the lowest is **{kpi_data['lowest_grade']:.0f}/20**.
            """
        )

    # Data Inspection Table
    with st.expander("📋 Inspect Filtered Dataset Records"):
        st.caption(
            "ℹ️ **Data Integrity Note:** The original UCI dataset is strictly anonymized and does not contain student names. "
            "**Anonymous Student IDs (STU-xxxx)** are synthetic sequence identifiers generated to index records. "
            "**School Institution** refers to the Portuguese public secondary school attended (GP or MS)."
        )
        inspect_df = filtered_df[
            [
                "student_id",
                "school",
                "school_name",
                "sex",
                "age",
                "studytime_label",
                "absences",
                "G1",
                "G2",
                "G3",
                "status",
                "grade_letter",
            ]
        ].copy()
        inspect_df.rename(
            columns={
                "student_id": "Anonymous Student ID",
                "school": "School Code",
                "school_name": "School Institution",
                "sex": "Sex",
                "age": "Age",
                "studytime_label": "Weekly Study Time",
                "absences": "Absence Count",
                "status": "Academic Status",
                "grade_letter": "Grade Band",
            },
            inplace=True,
        )
        st.dataframe(
            inspect_df,
            width="stretch",
            hide_index=True,
        )


# -----------------------------------------------------------------------------
# 2. Grade Progression (G1, G2, G3)
# -----------------------------------------------------------------------------
elif nav_selection == "📈 Grade Progression (G1/G2/G3)":
    st.title("📈 Academic Progression Across Evaluation Periods")
    st.markdown(
        """
        Evaluate how students evolve from **Term 1 (G1)** through **Term 2 (G2)** to the **Final Exam (G3)**. 
        Tracking longitudinal shifts helps educators detect early learning degradation or growth momentum.
        """
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric(
            label="Students Improved",
            value=f"{progression_data['improved_count']}",
            help=f"Final score G3 strictly greater than Term 1 score G1 ({progression_data['improved_pct']:.1f}% of cohort).",
        )
    with p2:
        st.metric(
            label="Students Declined",
            value=f"{progression_data['declined_count']}",
            help=f"Final score G3 strictly lower than Term 1 score G1 ({progression_data['declined_pct']:.1f}% of cohort).",
        )
    with p3:
        st.metric(
            label="Score Maintained",
            value=f"{progression_data['maintained_count']}",
            help=f"Final score G3 equal to Term 1 score G1 ({progression_data['maintained_pct']:.1f}% of cohort).",
        )
    with p4:
        st.metric(
            label="Net Average Trajectory",
            value=f"{progression_data['avg_delta']:+.2f} pts",
            help="Cohort-wide average delta: Final Exam (G3) minus Term 1 (G1).",
        )

    st.markdown("---")

    if not filtered_df.empty:
        fig_prog = plot_term_progression(filtered_df)
        st.pyplot(fig_prog, width="stretch")
        plt.close(fig_prog)

        st.markdown(
            '<div class="section-header">🌟 Top Academic Improvers (G3 vs. G1)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "ℹ️ **Data Integrity Note:** 'School Institution' indicates the Portuguese secondary high school attended "
            "(Escola Secundária Gabriel Pereira or Escola Secundária Mousinho da Silveira), not a student name. "
            "Student IDs are anonymous sequential identifiers."
        )
        top_improvers = filtered_df.sort_values(by="progression", ascending=False).head(5)
        top_improvers_display = top_improvers[
            [
                "student_id",
                "school_name",
                "G1",
                "G2",
                "G3",
                "progression",
                "studytime_label",
                "absences",
            ]
        ].copy()
        top_improvers_display.rename(
            columns={
                "student_id": "Anonymous Student ID",
                "school_name": "School Institution",
                "progression": "Score Growth (G3 - G1)",
                "studytime_label": "Weekly Study Time",
                "absences": "Absence Count",
            },
            inplace=True,
        )
        st.dataframe(
            top_improvers_display,
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("No records to display for the active filter selection.")


# -----------------------------------------------------------------------------
# 3. Attendance & Behavioral Factors
# -----------------------------------------------------------------------------
elif nav_selection == "🕒 Attendance & Behavior":
    st.title("🕒 Absence Patterns & Behavioral Dynamics")
    st.markdown(
        """
        Analyze how recorded **absence counts**, **weekly study commitment**, 
        and **lifestyle factors** statistically correlate with student examination outcomes.
        """
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric(
            label="Average Absence Count",
            value=f"{kpi_data['avg_absences']:.1f}",
            help="Mean recorded absences per student across the active cohort.",
        )
    with b2:
        high_abs_count = int((filtered_df["absences"] > 10).sum()) if not filtered_df.empty else 0
        high_abs_pct = (high_abs_count / len(filtered_df) * 100) if len(filtered_df) else 0.0
        st.metric(
            label="High Absences (>10)",
            value=f"{high_abs_count}",
            help=f"{high_abs_count} students ({high_abs_pct:.1f}% of cohort) have more than 10 recorded absences.",
        )
    with b3:
        if len(filtered_df) > 1:
            corr_val = filtered_df["absences"].corr(filtered_df["G3"])
            if pd.isna(corr_val):
                corr_val = 0.0
        else:
            corr_val = 0.0
        st.metric(
            label="Absences vs. Grade Correlation",
            value=f"{corr_val:.2f}",
            help="Pearson correlation coefficient (r) between recorded absence count and final exam score (G3).",
        )

    st.markdown("---")

    if filtered_df.empty:
        st.warning("No student records match the active filter criteria. Please broaden your sidebar filters.")
    else:
        col_att, col_study = st.columns(2)
        with col_att:
            st.markdown(
                '<div class="section-header">📉 Absence Count vs. Final Grade</div>',
                unsafe_allow_html=True,
            )
            fig_abs = plot_absences_vs_grades(filtered_df)
            st.pyplot(fig_abs, width="stretch")
            plt.close(fig_abs)

        with col_study:
            st.markdown(
                '<div class="section-header">📚 Study Time vs. Exam Results</div>',
                unsafe_allow_html=True,
            )
            fig_study = plot_study_time_impact(filtered_df)
            st.pyplot(fig_study, width="stretch")
            plt.close(fig_study)

        st.markdown("---")
        st.markdown(
            '<div class="section-header">🔥 Behavioral & Lifestyle Correlation Matrix</div>',
            unsafe_allow_html=True,
        )
        fig_heat = plot_correlation_heatmap(filtered_df)
        st.pyplot(fig_heat, width="stretch")
        plt.close(fig_heat)

        st.info(
            """
            **AI & Data Science Observations:**
            - **Strongest Predictors:** Preliminary examinations ($G1$ and $G2$) correlate strongest with the final grade ($r > 0.85$), proving early diagnostic checks are highly predictive.
            - **Negative Impact Factors:** Course failures ($failures$) and high absence counts show negative correlations with academic standing.
            - **Study Commitment:** Students reporting >= 5 hours of weekly study consistently achieve superior pass rates.
            """
        )


# -----------------------------------------------------------------------------
# 4. Student Profile & Search
# -----------------------------------------------------------------------------
elif nav_selection == "👤 Student Profile & Search":
    st.title("👤 Individual Student Diagnostic Profiler")
    st.markdown(
        """
        Inspect detailed individual performance metrics, student rankings, learning habits, 
        and tailored intervention suggestions for any student in the cohort.
        """
    )

    if filtered_df.empty:
        st.warning("No students available under current filters. Please adjust filters in the sidebar.")
    else:
        st.caption(
            "ℹ️ **Data Integrity Notice:** The original UCI dataset is strictly anonymized and does not contain student names. "
            "Student IDs (`STU-0001` through `STU-0395`) are synthetic anonymous identifiers generated for record indexing. "
            "School names represent public secondary schools in Portugal (GP and MS)."
        )

        student_options = filtered_df["student_id"].tolist()

        # Dual Search & Dropdown Selection Controls
        col_search, col_select = st.columns([1, 1.2])

        resolved_student_id = None
        with col_search:
            search_query = st.text_input(
                "🔍 Search by Anonymous ID:",
                placeholder="e.g. STU-0001 or 1",
                help="Type an ID like 'STU-0001' or simply '1' to jump to that student."
            ).strip().upper()

        if search_query:
            # Handle flexible input: '1' -> 'STU-0001', 'STU-1' -> 'STU-0001', 'STU-0001' -> 'STU-0001'
            if search_query.isdigit():
                candidate_id = f"STU-{int(search_query):04d}"
            elif search_query.startswith("STU-") and search_query[4:].isdigit():
                candidate_id = f"STU-{int(search_query[4:]):04d}"
            else:
                candidate_id = search_query

            if candidate_id in student_options:
                resolved_student_id = candidate_id
            else:
                if candidate_id in raw_df["student_id"].values:
                    st.warning(
                        f"⚠️ Student `{candidate_id}` exists in the dataset, but is excluded by your active cohort filters. "
                        f"Please broaden your sidebar filters to inspect this student."
                    )
                else:
                    st.error(
                        f"❌ Student ID `{search_query}` was not found. Valid IDs range from `STU-0001` to `STU-0395`."
                    )

        default_idx = 0
        if resolved_student_id and resolved_student_id in student_options:
            default_idx = student_options.index(resolved_student_id)

        with col_select:
            selected_student_id = st.selectbox(
                "Or Select from Active Cohort List:",
                options=student_options,
                index=default_idx,
                format_func=lambda sid: f"{sid} (Score: {filtered_df.loc[filtered_df['student_id'] == sid, 'G3'].values[0]}/20 - {filtered_df.loc[filtered_df['student_id'] == sid, 'status'].values[0]})",
            )

        active_id = resolved_student_id if resolved_student_id else selected_student_id
        profile = get_student_profile(filtered_df, active_id)

        if profile:
            rec = profile["record"]

            # Row 1: Term Examination Breakdown (G1, G2, G3, Progression)
            st.markdown('<div class="section-header">📝 Examination Grades & Progression</div>', unsafe_allow_html=True)
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.metric(
                    label="Term 1 Exam (G1)",
                    value=f"{rec['G1']} / 20",
                    help=f"First term score: {rec['G1']}/20 ({rec['G1_percentage']:.1f}% equivalent)."
                )
            with g2:
                st.metric(
                    label="Term 2 Exam (G2)",
                    value=f"{rec['G2']} / 20",
                    help=f"Second term score: {rec['G2']}/20 ({rec['G2_percentage']:.1f}% equivalent)."
                )
            with g3:
                st.metric(
                    label="Final Exam (G3)",
                    value=f"{rec['G3']} / 20",
                    help=f"Final score: {rec['G3']}/20 ({rec['final_percentage']:.1f}%) | Band: {rec['grade_band']}."
                )
            with g4:
                st.metric(
                    label="Score Trajectory (G3 - G1)",
                    value=f"{rec['progression']:+d} pts",
                    help=f"Point growth or drop between Term 1 and Final Exam ({rec['progression_type']})."
                )

            # Row 2: Standing, Absences, Study Time, Status
            st.markdown('<div class="section-header">📊 Cohort Standing & Learning Habits</div>', unsafe_allow_html=True)
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.metric(
                    label="Class Standing",
                    value=f"Rank #{profile['rank']} of {len(filtered_df)}",
                    help=f"Student ranks at the {profile['percentile']}th percentile of the cohort."
                )
            with h2:
                st.metric(
                    label="Academic Standing",
                    value=rec["status"],
                    help=f"Grade Band: {rec['grade_band']}."
                )
            with h3:
                st.metric(
                    label="Recorded Absences",
                    value=f"{rec['absences']}",
                    help=f"Absence Tier: {rec['absence_tier']} (Cohort average is {profile['class_averages']['absences']:.1f})."
                )
            with h4:
                st.metric(
                    label="Weekly Study Time",
                    value=rec["studytime_label"],
                    help="Reported time spent on self-study outside regular classes."
                )

            st.markdown("---")

            # Comparison and Characteristics
            col_chart, col_details = st.columns([1.1, 0.9])

            with col_chart:
                st.markdown(
                    f'<div class="section-header">📊 {rec["student_id"]} vs. Class Average</div>',
                    unsafe_allow_html=True,
                )
                fig_comp = plot_student_vs_class_bar(rec, profile["class_averages"])
                st.pyplot(fig_comp, width="stretch")
                plt.close(fig_comp)

            with col_details:
                st.markdown(
                    '<div class="section-header">📋 Student Information & Demographics</div>',
                    unsafe_allow_html=True,
                )
                st.write(f"**Enrolled Institution:** {rec['school_name']}")
                st.caption("ℹ️ 'Gabriel Pereira' and 'Mousinho da Silveira' are Portuguese public high schools, not student names.")
                st.write(f"**Gender / Age:** {'Female' if rec['sex'] == 'F' else 'Male'}, {rec['age']} years old")
                st.write(f"**Address:** {rec['address_type']}")
                st.write(f"**Guardian:** {rec['guardian'].capitalize()}")
                st.write(f"**Family Support:** {'Yes' if rec['famsup'] == 'yes' else 'No'}")
                st.write(f"**Extra Tutoring:** {'Yes' if rec['schoolsup'] == 'yes' else 'No'}")
                st.write(f"**Internet at Home:** {'Yes' if rec['internet'] == 'yes' else 'No'}")
                st.write(f"**Past Course Failures:** {rec['failures']}")

            # Automated Diagnostic Strengths & Concerns
            st.markdown(
                '<div class="section-header">🩺 Diagnostic Feedback & Intervention Recommendations</div>',
                unsafe_allow_html=True,
            )
            col_str, col_con = st.columns(2)

            with col_str:
                st.subheader("🟢 Strengths & Positive Indicators")
                if profile["strengths"]:
                    for s in profile["strengths"]:
                        st.success(f"✔ {s}")
                else:
                    st.info("No standout strengths recorded.")

            with col_con:
                st.subheader("🔴 Concerns & Risk Factors")
                if profile["concerns"]:
                    for c in profile["concerns"]:
                        st.error(f"⚠ {c}")
                else:
                    st.success("🎉 No pressing academic concerns identified.")

            # Download Report Card
            st.markdown("---")
            report_card_text = generate_student_report_card(profile)
            st.download_button(
                label=f"📥 Download Report Card for {rec['student_id']} (Anonymous)",
                data=report_card_text,
                file_name=f"Report_Card_{rec['student_id']}.txt",
                mime="text/plain",
            )


# -----------------------------------------------------------------------------
# 5. At-Risk Early Warning System & Machine Learning Forecasting
# -----------------------------------------------------------------------------
elif nav_selection == "⚠️ At-Risk Early Warning & ML Forecasting":
    st.title("⚠️ Academic Early Warning & ML Forecasting System")
    st.markdown(
        """
        Identify students requiring academic assistance through **transparent rule-based criteria** 
        and **explainable Machine Learning predictive models** (Random Forest classification & regression).
        """
    )

    tab_rules, tab_ml = st.tabs(["📋 Rule-Based Early Warning", "🤖 Machine Learning Predictive Model"])

    with tab_rules:
        # Custom Threshold Controllers
        st.markdown('<div class="section-header">⚙️ Custom Risk Thresholds</div>', unsafe_allow_html=True)
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            threshold_grade = st.slider(
                "Minimum Passing Grade Cutoff (0-20):",
                min_value=5.0,
                max_value=14.0,
                value=10.0,
                step=0.5,
                help="Students with final or continuous grades below this value are prioritized.",
            )
        with ctrl_col2:
            threshold_absences = st.slider(
                "Absence Cutoff Threshold:",
                min_value=4,
                max_value=25,
                value=10,
                step=1,
                help="Students with absences exceeding this number are flagged.",
            )

        # Identify At-Risk Students
        at_risk_df = identify_at_risk_students(
            filtered_df,
            min_pass_grade=threshold_grade,
            max_allowed_absences=threshold_absences,
        )

        # Risk Metrics (No misleading delta arrows)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric(
                label="Total Flagged Students",
                value=f"{len(at_risk_df):,}",
                help="Total students meeting at least one risk factor.",
            )
        with r2:
            risk_pct = (len(at_risk_df) / len(filtered_df) * 100) if len(filtered_df) else 0.0
            st.metric(
                label="Proportion of Cohort",
                value=f"{risk_pct:.1f}%",
                help="Percentage of the active filtered cohort identified as at-risk.",
            )
        with r3:
            high_risk_count = int((at_risk_df["risk_level"] == "High Risk").sum()) if not at_risk_df.empty else 0
            st.metric(
                label="High-Risk Urgency Cases",
                value=f"{high_risk_count}",
                help="High-risk students with failing marks combined with chronic absences or prior course failures.",
            )

        st.markdown("---")

        if not at_risk_df.empty:
            st.markdown(
                '<div class="section-header">📋 At-Risk Student Registry & Diagnosed Reasons</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "ℹ️ **Data Note:** 'Anonymous Student ID' is generated for record tracking. "
                "'School Institution' denotes the secondary high school attended in Portugal (GP or MS)."
            )
            at_risk_display = at_risk_df[
                [
                    "student_id",
                    "risk_level",
                    "risk_reasons",
                    "G1",
                    "G2",
                    "G3",
                    "absences",
                    "failures",
                    "studytime_label",
                    "school_name",
                ]
            ].copy()
            at_risk_display.rename(
                columns={
                    "student_id": "Anonymous Student ID",
                    "risk_level": "Risk Category",
                    "risk_reasons": "Identified Risk Triggers",
                    "absences": "Absence Count",
                    "failures": "Past Failures",
                    "studytime_label": "Weekly Study Time",
                    "school_name": "School Institution",
                },
                inplace=True,
            )
            st.dataframe(
                at_risk_display,
                width="stretch",
                hide_index=True,
            )

            # Download At-Risk CSV
            csv_bytes = export_at_risk_report_csv(at_risk_df)
            st.download_button(
                label="📥 Export At-Risk Registry (CSV)",
                data=csv_bytes,
                file_name="at_risk_students_registry.csv",
                mime="text/csv",
            )
        else:
            st.success("🎉 Excellent! No students in the current filtered selection meet the at-risk criteria.")

    with tab_ml:
        st.markdown('<div class="section-header">🤖 Supervised Machine Learning Outcome Forecasting</div>', unsafe_allow_html=True)
        st.write(
            "Predict academic success using Scikit-learn **Random Forest Classifier** (Pass/Fail) "
            "and **Random Forest Regressor** (Final Score G3), trained on early-semester performance indicators."
        )

        ml_cohort = filtered_df if len(filtered_df) >= 30 else raw_df
        if len(filtered_df) < 30:
            st.info("ℹ️ Filtered cohort has fewer than 30 students. The ML model was trained on the entire benchmark cohort.")

        model_bundle = train_performance_models(ml_cohort)

        if not model_bundle.get("trained", False):
            st.warning(f"⚠️ Model training unavailable: {model_bundle.get('reason')}")
        else:
            metrics = model_bundle["metrics"]

            # Out-of-sample Test Evaluation Cards
            st.markdown('<div class="section-header">🎯 Out-of-Sample Test Evaluation (80/20 Holdout)</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    "Classifier Accuracy",
                    f"{metrics['accuracy']:.1f}%",
                    help="Holdout test accuracy of Random Forest predicting Pass/Fail (cutoff >= 10/20)."
                )
            with m2:
                st.metric(
                    "Weighted F1-Score",
                    f"{metrics['f1_score']:.2f}",
                    help="Harmonic mean of precision and recall accounting for class balance."
                )
            with m3:
                st.metric(
                    "Regression R² Score",
                    f"{metrics['r2_score']:.2f}",
                    help="Coefficient of determination for predicting final continuous G3 grade."
                )
            with m4:
                st.metric(
                    "Forecast RMSE",
                    f"{metrics['rmse']:.2f} pts",
                    help="Root Mean Squared Error of final grade predictions on test set."
                )

            st.markdown("---")

            # Feature Importances & Architecture Notes
            col_imp, col_arch = st.columns([1.1, 0.9])
            with col_imp:
                st.markdown('<div class="section-header">📊 Top Predictive Academic Drivers</div>', unsafe_allow_html=True)
                fig_imp = plot_feature_importance(model_bundle["feature_importances"])
                st.pyplot(fig_imp, width="stretch")
                plt.close(fig_imp)

            with col_arch:
                st.markdown('<div class="section-header">🔬 Machine Learning Architecture</div>', unsafe_allow_html=True)
                st.markdown(
                    r"""
                    - **Classification Model:** Random Forest Classifier (`n_estimators=100`, `max_depth=5`) predicting semester outcome (**Pass** $\ge 10$ vs. **Fail / At-Risk** $< 10$).
                    - **Regression Model:** Random Forest Regressor (`n_estimators=100`, `max_depth=5`) estimating exact continuous $G3$ final exam mark ($0-20$).
                    - **Validation Method:** Stratified holdout test split ($80\%$ train, $20\%$ holdout test) with fixed seed for reproducibility.
                    - **Predictive Factors:** Historical grades ($G1, G2$), attendance habits, past academic failures, weekly study time, and home environment.
                    - **Data Integrity:** Predictions are strictly tagged as statistical forecasts for academic intervention planning.
                    """
                )

            st.markdown("---")

            # Interactive What-If Student Outcome Simulator
            st.markdown('<div class="section-header">🔮 Interactive "What-If" Student Outcome Simulator</div>', unsafe_allow_html=True)
            st.caption("Adjust student metrics below to simulate forecasted final outcomes and evaluate the impact of targeted academic interventions.")

            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                sim_g1 = st.slider("Term 1 Score (G1, 0-20):", min_value=0.0, max_value=20.0, value=11.0, step=0.5)
                sim_g2 = st.slider("Term 2 Score (G2, 0-20):", min_value=0.0, max_value=20.0, value=11.0, step=0.5)
            with sim_c2:
                sim_studytime = st.selectbox(
                    "Weekly Study Time:",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {1: "< 2 hours", 2: "2 - 5 hours", 3: "5 - 10 hours", 4: "> 10 hours"}[x],
                    index=1,
                )
                sim_absences = st.slider("Recorded Absences:", min_value=0, max_value=50, value=4, step=1)
            with sim_c3:
                sim_failures = st.selectbox("Past Course Failures:", options=[0, 1, 2, 3, 4], index=0)
                sim_medu = st.selectbox(
                    "Mother's Education (Medu):",
                    options=[0, 1, 2, 3, 4],
                    format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th to 9th grade", 3: "Secondary", 4: "Higher Education"}[x],
                    index=2,
                )

            sim_record = {
                "G1": sim_g1,
                "G2": sim_g2,
                "studytime": sim_studytime,
                "absences": sim_absences,
                "failures": sim_failures,
                "Medu": sim_medu,
                "Fedu": sim_medu,
                "traveltime": 1,
                "famrel": 4,
                "freetime": 3,
                "goout": 3,
                "health": 4,
            }

            sim_result = predict_single_student(model_bundle, sim_record)

            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                status_label = sim_result["predicted_status"]
                if sim_result["predicted_pass"]:
                    st.success(f"### {status_label}")
                else:
                    st.error(f"### {status_label}")
            with res_c2:
                st.metric("Forecasted Pass Probability", f"{sim_result['pass_probability']:.1f}%", help="Confidence of achieving >= 10/20 passing grade.")
            with res_c3:
                st.metric("Forecasted Final Score (G3)", f"{sim_result['predicted_G3_score']:.1f} / 20", help="Continuous final score projection from Random Forest Regressor.")

            st.info(f"**Forecast Confidence Level:** {sim_result['confidence']} • **Key Drivers Identified:** " + " | ".join(sim_result["key_drivers"]))
            st.caption("ℹ️ *Disclaimer: Machine learning predictions are probabilistic forecasts designed to assist advisors with timely interventions and do not determine official student grades.*")

            st.markdown("---")

            # Full Cohort Predictions Table
            st.markdown('<div class="section-header">📑 Cohort Forecast Registry & Probabilities</div>', unsafe_allow_html=True)
            st.caption("Complete tabular view of ML predictions and pass probabilities for the active cohort:")
            preds_display = model_bundle["predictions_df"][[
                "student_id", "school_name", "G1", "G2", "G3", "status",
                "predicted_status", "predicted_pass_prob", "predicted_G3"
            ]].copy()
            preds_display.rename(columns={
                "student_id": "Anonymous Student ID",
                "school_name": "School Institution",
                "status": "Actual Standing",
                "predicted_status": "ML Forecast Standing",
                "predicted_pass_prob": "Forecast Pass Prob (%)",
                "predicted_G3": "Forecast G3 Score"
            }, inplace=True)
            st.dataframe(preds_display, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# 6. Export Reports & Data
# -----------------------------------------------------------------------------
elif nav_selection == "📥 Export Reports & Data":
    st.title("📥 Reporting & Data Export Center")
    st.markdown(
        """
        Download comprehensive executive summaries, filtered CSV datasets, and explore the complete dataset dictionary.
        """
    )

    t1, t2, t3 = st.tabs(["📄 Executive Summary", "📊 Export Datasets", "📖 Dataset Dictionary"])

    with t1:
        st.subheader("Cohort Performance Executive Summary")
        summary_text = generate_executive_summary_text(filtered_df, kpi_data, progression_data)
        st.text_area("Executive Summary Preview:", value=summary_text, height=350)

        st.download_button(
            label="📥 Download Executive Summary (TXT)",
            data=summary_text,
            file_name="Cohort_Executive_Summary.txt",
            mime="text/plain",
        )

    with t2:
        st.subheader("Export Tabular Data (CSV)")
        st.write("Download the active, filtered dataset including all derived metrics and student IDs.")
        filtered_csv = export_dataframe_to_csv(filtered_df)

        st.download_button(
            label="📥 Download Filtered Dataset (CSV)",
            data=filtered_csv,
            file_name="filtered_student_analytics.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.write("Download the complete at-risk registry for academic advisor follow-up.")
        at_risk_data = identify_at_risk_students(filtered_df)
        at_risk_csv = export_at_risk_report_csv(at_risk_data)

        st.download_button(
            label="📥 Download At-Risk Registry (CSV)",
            data=at_risk_csv,
            file_name="at_risk_students_report.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.write("Download machine learning outcome forecasts with probability scores and predicted final grades.")
        ml_bundle_export = train_performance_models(filtered_df if len(filtered_df) >= 30 else raw_df)
        if ml_bundle_export.get("trained", False):
            ml_preds_csv = export_dataframe_to_csv(ml_bundle_export["predictions_df"])
            st.download_button(
                label="📥 Download ML Predictions Dataset (CSV)",
                data=ml_preds_csv,
                file_name="ml_predicted_student_outcomes.csv",
                mime="text/csv",
            )

    with t3:
        st.subheader("UCI Student Performance Dataset Dictionary")
        st.write("Detailed metadata reference for all 33 variables in `student-mat.csv`:")
        meta = get_column_metadata()
        meta_df = pd.DataFrame.from_dict(meta, orient="index").reset_index()
        meta_df.columns = ["Variable Name", "Description", "Data Type", "Accepted Values / Scale"]
        st.dataframe(meta_df, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    Student Performance & Learning Analytics System • Built with Python, Pandas, NumPy, Matplotlib, Seaborn & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
