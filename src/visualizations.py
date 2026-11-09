"""
visualizations.py
Module providing publication-quality statistical charts built with Matplotlib and Seaborn
for the Student Performance & Learning Analytics System.
"""

from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for web apps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Consistent styling configuration
THEME_PALETTE = {
    "primary": "#2563EB",     # Modern Royal Blue
    "secondary": "#10B981",   # Emerald Green (Pass)
    "danger": "#EF4444",      # Vibrant Red (Fail / Risk)
    "warning": "#F59E0B",     # Amber (Moderate / Watchlist)
    "dark": "#1E293B",        # Slate Dark
    "accent": "#8B5CF6"       # Violet
}

# Set Seaborn baseline styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "axes.edgecolor": "#CBD5E1",
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})


def plot_grade_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Renders the final grade (G3) distribution with KDE curve, mean line, and pass threshold.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    # Plot histogram with KDE
    sns.histplot(
        df["G3"],
        bins=21,
        binrange=(0, 20),
        kde=True,
        color=THEME_PALETTE["primary"],
        edgecolor="#FFFFFF",
        alpha=0.65,
        ax=ax
    )

    # Vertical reference lines
    mean_val = df["G3"].mean()
    ax.axvline(
        x=mean_val,
        color=THEME_PALETTE["dark"],
        linestyle="--",
        linewidth=2,
        label=f"Class Mean: {mean_val:.2f} / 20"
    )
    ax.axvline(
        x=10,
        color=THEME_PALETTE["danger"],
        linestyle=":",
        linewidth=2.5,
        label="Passing Cutoff (10 / 20)"
    )

    ax.set_title("Distribution of Final Exam Grades (G3)", fontweight="bold", pad=12)
    ax.set_xlabel("Final Grade (0 to 20 Scale)")
    ax.set_ylabel("Student Count")
    ax.set_xlim(-0.5, 20.5)
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    return fig


def plot_term_progression(df: pd.DataFrame) -> plt.Figure:
    """
    Compares student performance distributions across Term 1 (G1), Term 2 (G2), and Final (G3).
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=100)

    terms_df = df[["G1", "G2", "G3"]].melt(
        var_name="Term",
        value_name="Score"
    )
    terms_df["Term"] = terms_df["Term"].map({
        "G1": "Term 1 (G1)",
        "G2": "Term 2 (G2)",
        "G3": "Final Exam (G3)"
    })

    palette = [THEME_PALETTE["accent"], THEME_PALETTE["primary"], THEME_PALETTE["secondary"]]
    
    sns.boxplot(
        data=terms_df,
        x="Term",
        y="Score",
        hue="Term",
        palette=palette,
        legend=False,
        width=0.45,
        fliersize=3,
        boxprops=dict(alpha=0.85),
        ax=ax
    )

    # Add means as points with text
    means = [df["G1"].mean(), df["G2"].mean(), df["G3"].mean()]
    for i, m in enumerate(means):
        ax.scatter(i, m, color="#FFFFFF", edgecolor=THEME_PALETTE["dark"], s=60, zorder=5)
        ax.annotate(
            f"Avg: {m:.2f}",
            (i, m),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="#CBD5E1", alpha=0.9)
        )

    ax.axhline(10, color=THEME_PALETTE["danger"], linestyle=":", linewidth=1.5, label="Passing Mark (10/20)")
    ax.set_title("Academic Performance Progression Across Terms", fontweight="bold", pad=12)
    ax.set_xlabel("Evaluation Period")
    ax.set_ylabel("Score (0 to 20)")
    ax.set_ylim(-1, 22)
    ax.legend(loc="lower right", frameon=True, facecolor="#F8FAFC")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    return fig


def plot_absences_vs_grades(df: pd.DataFrame) -> plt.Figure:
    """
    Renders a scatter plot with regression trendline illustrating the impact of absences on final grades.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=100)

    # Pearson correlation with fallback for constant/single records
    corr = df["absences"].corr(df["G3"]) if len(df) > 1 else 0.0
    if pd.isna(corr):
        corr = 0.0

    # Scatter plot segregated by passing status
    sns.scatterplot(
        data=df,
        x="absences",
        y="G3",
        hue="status",
        palette={"Passed": THEME_PALETTE["secondary"], "Failed": THEME_PALETTE["danger"]},
        alpha=0.75,
        s=60,
        edgecolor="#FFFFFF",
        ax=ax
    )

    # Linear trendline
    if len(df) > 2:
        sns.regplot(
            data=df,
            x="absences",
            y="G3",
            scatter=False,
            ax=ax,
            color=THEME_PALETTE["dark"],
            line_kws={"linestyle": "--", "linewidth": 2, "label": f"Trendline (r = {corr:.2f})"}
        )

    ax.set_title(f"Absence Count vs. Final Grade (Correlation r = {corr:.2f})", fontweight="bold", pad=12)
    ax.set_xlabel("Recorded Absences (Count)")
    ax.set_ylabel("Final Grade (G3)")
    ax.set_ylim(-1, 21)
    ax.axhline(10, color=THEME_PALETTE["danger"], linestyle=":", linewidth=1.2, alpha=0.7)
    ax.legend(loc="upper right", frameon=True, facecolor="#F8FAFC")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    return fig


def plot_study_time_impact(df: pd.DataFrame) -> plt.Figure:
    """
    Bar plot showing average final grade achieved across weekly study time tiers.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=100)

    # Group by study time label in order
    order = ["< 2 hours", "2 - 5 hours", "5 - 10 hours", "> 10 hours"]
    study_stats = df.groupby("studytime_label", as_index=False)["G3"].agg(["mean", "std", "count"])
    study_stats = study_stats.reindex(order).fillna(0.0)

    # Safe error calculation
    y_err = []
    for _, s_row in study_stats.iterrows():
        cnt = s_row["count"]
        sd = s_row["std"]
        if cnt > 1 and sd > 0:
            y_err.append(sd / np.sqrt(cnt))
        else:
            y_err.append(0.0)

    bars = ax.bar(
        order,
        study_stats["mean"],
        yerr=y_err,
        capsize=4,
        color=THEME_PALETTE["primary"],
        alpha=0.85,
        edgecolor="#FFFFFF"
    )

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.2f}",
                (bar.get_x() + bar.get_width() / 2, h / 2),
                ha="center",
                va="center",
                color="#FFFFFF",
                fontweight="bold",
                fontsize=10
            )

    ax.axhline(10, color=THEME_PALETTE["danger"], linestyle=":", linewidth=1.5, label="Passing Mark (10)")
    ax.set_title("Impact of Weekly Study Time on Final Grade", fontweight="bold", pad=12)
    ax.set_xlabel("Weekly Study Time Category")
    ax.set_ylabel("Average Final Grade (G3)")
    ax.set_ylim(0, 20)
    ax.legend(loc="lower right", frameon=True)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Renders a styled heatmap of correlations among academic, behavioral, and personal factors.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=100)

    features = [
        "studytime", "failures", "famrel", "freetime",
        "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"
    ]
    labels = [
        "Study Time", "Failures", "Family Rel", "Free Time",
        "Go Out", "Workday Alc", "Weekend Alc", "Health", "Absences", "Term 1", "Term 2", "Final (G3)"
    ]

    corr = df[features].corr().fillna(0.0)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.4,
        center=0,
        square=True,
        linewidths=0.75,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_title("Correlation Heatmap: Academics, Habits & Lifestyle", fontweight="bold", pad=14)
    plt.tight_layout()
    return fig


def plot_grade_breakdown_donut(df: pd.DataFrame) -> plt.Figure:
    """
    Renders a donut chart illustrating student letter grade distribution.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)

    grade_order = ["A", "B", "C", "D", "F"]
    counts = df["grade_letter"].value_counts().reindex(grade_order, fill_value=0)

    colors = [
        "#10B981",  # A: Emerald
        "#3B82F6",  # B: Blue
        "#8B5CF6",  # C: Purple
        "#F59E0B",  # D: Amber
        "#EF4444"   # F: Red
    ]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[f"Grade {g}" for g in grade_order],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="#FFFFFF", linewidth=2)
    )

    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")

    ax.set_title("Student Grade Band Breakdown", fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_student_vs_class_bar(
    student_record: pd.Series,
    class_averages: Dict[str, float]
) -> plt.Figure:
    """
    Compares an individual student's key academic indicators against the class average.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)

    categories = ["Term 1 (G1)", "Term 2 (G2)", "Final (G3)", "Absences"]
    student_vals = [
        student_record["G1"],
        student_record["G2"],
        student_record["G3"],
        student_record["absences"]
    ]
    class_vals = [
        class_averages["G1"],
        class_averages["G2"],
        class_averages["G3"],
        class_averages["absences"]
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, student_vals, width, label="Student", color=THEME_PALETTE["primary"])
    bars2 = ax.bar(x + width/2, class_vals, width, label="Class Average", color="#94A3B8")

    # Add value annotations
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)

    ax.set_title(f"Comparison: {student_record['student_id']} vs. Class Average", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score (0-20) / Absence Count")
    ax.legend(frameon=True, facecolor="#F8FAFC")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_df: pd.DataFrame) -> plt.Figure:
    """
    Renders a clean horizontal bar chart displaying machine learning feature importances.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    top_features = feature_df.head(8).sort_values(by="importance", ascending=True)

    bars = ax.barh(
        top_features["feature_name"],
        top_features["importance"],
        color=THEME_PALETTE["primary"],
        alpha=0.85,
        edgecolor="#FFFFFF"
    )

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w * 100:.1f}%",
            (w, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontweight="bold",
            fontsize=8.5,
            color=THEME_PALETTE["dark"]
        )

    ax.set_title("Top Predictive Factors (Random Forest ML Model)", fontweight="bold", pad=12)
    ax.set_xlabel("Relative Feature Importance Weight")
    ax.set_xlim(0, max(top_features["importance"]) * 1.25 if not top_features.empty else 1.0)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    return fig

