"""Clinical Trial Data Analysis and Automation Dashboard."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 1000
REFRESH_SECONDS = 60

ROOT = Path(__file__).resolve().parent.parent

LIVE_SNAPSHOT_PATH = (
    ROOT / "data" / "processed"
    / "clinicaltrials_live_dashboard.csv"
)
MASTER_DATA_PATH = (
    ROOT / "data" / "processed"
    / "clinical_trials_master.csv"
)
VALIDATION_ISSUES_PATH = (
    ROOT / "reports"
    / "pipeline_validation_issues.csv"
)

st.set_page_config(
    page_title="Clinical Trial Dashboard",
    layout="wide",
)

st.title("Clinical Trial Data Analysis & Automation")
st.caption(
    "Live data from ClinicalTrials.gov, with a separate "
    "pipeline validation summary."
)


def fetch_live_trials():
    """Fetch the latest published studies from the API."""

    params = {
        "format": "json",
        "pageSize": PAGE_SIZE,
        "countTotal": "true",
        "sort": "LastUpdatePostDate:desc",
    }

    headers = {
        "User-Agent": "ClinicalTrialAnalyticsDashboard/1.0",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=headers,
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()

    rows = []

    for study in payload.get("studies", []):
        protocol = study.get("protocolSection", {})

        ident = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        sponsor_module = protocol.get(
            "sponsorCollaboratorsModule", {}
        )
        conditions_module = protocol.get(
            "conditionsModule", {}
        )
        locations_module = protocol.get(
            "contactsLocationsModule", {}
        )

        locations = locations_module.get("locations", []) or []

        countries = sorted({
            location.get("country", "").strip()
            for location in locations
            if isinstance(location, dict)
            and location.get("country", "").strip()
        })

        start = status.get("startDateStruct", {}) or {}
        completion = status.get(
            "completionDateStruct", {}
        ) or {}

        last_update = (
            status.get("studyLastUpdatePostDateStruct", {})
            or status.get("lastUpdatePostDateStruct", {})
            or {}
        )

        sponsor = sponsor_module.get("leadSponsor", {}) or {}
        enrollment = design.get("enrollmentInfo", {}) or {}
        nct_id = ident.get("nctId", "")

        rows.append({
            "NCT ID": nct_id,
            "Title": ident.get("briefTitle", ""),
            "Status": status.get("overallStatus", "Unknown"),
            "Study type": design.get(
                "studyType", "Not specified"
            ),
            "Phase": ", ".join(
                design.get("phases", []) or []
            ) or "Not specified",
            "Sponsor": sponsor.get("name", "Unknown"),
            "Conditions": "; ".join(
                conditions_module.get("conditions", []) or []
            ),
            "Start date": start.get("date", ""),
            "Completion date": completion.get("date", ""),
            "Last update posted": last_update.get("date", ""),
            "Countries": ", ".join(countries) or "Not specified",
            "Enrollment": enrollment.get("count"),
            "Study URL": (
                f"https://clinicaltrials.gov/study/{nct_id}"
                if nct_id else ""
            ),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(
            subset=["NCT ID"],
            keep="first",
        ).reset_index(drop=True)

    return (
        df,
        payload.get("totalCount"),
        datetime.now().astimezone(),
    )


def save_live_snapshot(df):
    """Save live records separately from pipeline files."""

    LIVE_SNAPSHOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(LIVE_SNAPSHOT_PATH, index=False)
    return LIVE_SNAPSHOT_PATH


def calculate_live_quality(df):
    """Calculate completeness of 12 live-data fields."""

    fields = [
        "NCT ID",
        "Title",
        "Status",
        "Study type",
        "Phase",
        "Sponsor",
        "Conditions",
        "Start date",
        "Completion date",
        "Last update posted",
        "Countries",
        "Enrollment",
    ]

    if df.empty:
        return 0.0

    quality_df = df[fields].copy()

    for column in quality_df.columns:
        quality_df[column] = quality_df[column].apply(
            lambda value: (
                pd.notna(value)
                and str(value).strip().lower()
                not in {"", "not specified", "unknown", "nan"}
            )
        )

    return round(
        100 * quality_df.to_numpy().mean(),
        2,
    )


def find_column(df, candidates):
    """Find a column even if its capitalization differs."""

    lookup = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]

    return None

def read_csv_safe(path):
    """
    Read CSV files without converting literal 'NA' values
    into missing values.

    Empty CSV fields are still treated as missing values.
    """
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )


def load_pipeline_validation():
    """
    Read saved pipeline outputs.
    This function never runs the pipeline or writes its reports.
    """

    master = pd.DataFrame()
    issues = pd.DataFrame()
    master_error = None
    issues_error = None

    if MASTER_DATA_PATH.exists():
        try:
            master = read_csv_safe(MASTER_DATA_PATH)
        except Exception as exc:
            master_error = str(exc)
    else:
        master_error = f"File not found: {MASTER_DATA_PATH}"

    if VALIDATION_ISSUES_PATH.exists():
        try:
            issues = read_csv_safe(VALIDATION_ISSUES_PATH)
        except Exception as exc:
            issues_error = str(exc)
    else:
        issues_error = f"File not found: {VALIDATION_ISSUES_PATH}"

    if not master.empty:
        id_col = find_column(
            master,
            ["nct_id", "study_id", "NCT ID", "nctid"],
        )

        if id_col:
            master["_study_key"] = (
                master[id_col].astype(str).str.strip().str.upper()
            )

    if not issues.empty:
        issue_id_col = find_column(
            issues,
            ["nct_id", "study_id", "NCT ID", "nctid", "studyid"],
        )

        if issue_id_col:
            issues["_study_key"] = (
                issues[issue_id_col]
                .astype(str)
                .str.strip()
                .str.upper()
            )
        else:
            issues["_study_key"] = ""

        description_col = find_column(
            issues,
            [
                "issue_description",
                "description",
                "issue_type",
                "issue",
                "message",
                "validation_issue",
                "rule",
            ],
        )

        if description_col:
            issues["Issue description"] = (
                issues[description_col]
                .fillna("")
                .astype(str)
            )
        else:
            excluded = {"_study_key"}
            if issue_id_col:
                excluded.add(issue_id_col)

            visible_columns = [
                column for column in issues.columns
                if column not in excluded
            ]

            if visible_columns:
                issues["Issue description"] = issues[
                    visible_columns
                ].apply(
                    lambda row: " | ".join(
                        f"{column}: {row[column]}"
                        for column in visible_columns
                        if pd.notna(row[column])
                        and str(row[column]).strip()
                    ),
                    axis=1,
                )
            else:
                issues["Issue description"] = (
                    "Validation issue recorded"
                )

    return master, issues, master_error, issues_error

def build_validation_table(master, issues):
    """Combine study validation results with issue descriptions."""

    if master.empty:
        return pd.DataFrame()

    id_col = find_column(
        master,
        ["nct_id", "study_id", "NCT ID", "nctid"],
    )

    if not id_col:
        return pd.DataFrame()

    status_col = find_column(
        master,
        ["validation_status", "validation status"],
    )
    quality_col = find_column(
        master,
        ["quality_status", "quality status"],
    )
    score_col = find_column(
        master,
        ["quality_score", "quality score"],
    )
    title_col = find_column(
        master,
        ["brief_title", "title", "study_title"],
    )

    result = pd.DataFrame({
        "NCT ID": master[id_col].astype(str).str.strip(),
    })

    result["_study_key"] = result["NCT ID"].str.upper()

    result["Title"] = (
        master[title_col].fillna("").astype(str)
        if title_col else ""
    )

    result["Validation status"] = (
        master[status_col].fillna("Unknown").astype(str)
        if status_col else "Unknown"
    )

    result["Quality status"] = (
        master[quality_col].fillna("Unknown").astype(str)
        if quality_col else "Unknown"
    )

    result["Quality score"] = (
        pd.to_numeric(master[score_col], errors="coerce")
        if score_col else pd.NA
    )

    issue_map = {}

    if (
        not issues.empty
        and "_study_key" in issues.columns
        and "Issue description" in issues.columns
    ):
        valid_issues = issues[
            issues["_study_key"].ne("")
        ]

        issue_map = (
            valid_issues.groupby("_study_key")["Issue description"]
            .apply(
                lambda values: "; ".join(dict.fromkeys(
                    str(value).strip()
                    for value in values
                    if pd.notna(value) and str(value).strip()
                ))
            )
            .to_dict()
        )

    result["Issue descriptions"] = (
        result["_study_key"].map(issue_map).fillna("")
    )

    review_mask = (
        result["Validation status"]
        .str.upper()
        .eq("REVIEW_REQUIRED")
    )

    no_description = (
        review_mask & result["Issue descriptions"].eq("")
    )

    result.loc[
        no_description, "Issue descriptions"
    ] = (
        "Marked REVIEW_REQUIRED, but no matching issue detail "
        "was found in the saved issues report."
    )

    result = result.drop(columns=["_study_key"])

    return result


# Optional automatic refresh every 60 seconds.
try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(
        interval=REFRESH_SECONDS * 1000,
        key="clinical_trial_auto_refresh",
    )
    auto_refresh_available = True

except ImportError:
    auto_refresh_available = False

if not auto_refresh_available:
    st.info(
        "Automatic refresh is not installed. To enable it, run: "
        "python -m pip install streamlit-autorefresh"
    )

if st.button("🔄 Refresh live data now", type="primary"):
    st.rerun()


# Fetch current API data.
try:
    with st.spinner(
        "Fetching the latest studies from ClinicalTrials.gov..."
    ):
        df, total_available, fetched_at = fetch_live_trials()

except Exception as exc:
    st.error(f"Live API request failed: {exc}")
    st.warning(
        "No old dataset is being substituted for the live response. "
        "Check your internet connection and try again."
    )
    st.stop()

if df.empty:
    st.warning(
        "The API returned no records. No old data is displayed."
    )
    st.stop()

st.success(
    f"Live API request succeeded: {len(df):,} records received."
)

registry_total = (
    f"{total_available:,}"
    if isinstance(total_available, int)
    else str(total_available or "Unavailable")
)

st.caption(
    f"Last successful fetch: "
    f"{fetched_at.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
    f"Registry total: {registry_total} | "
    f"Refresh interval: {REFRESH_SECONDS} seconds"
)


# Save only to a separate dashboard snapshot.
try:
    save_live_snapshot(df)
    st.success(
        f"Live snapshot saved separately "
        f"({len(df):,} records). Pipeline files were not overwritten."
    )
except Exception as exc:
    st.warning(f"Could not save live snapshot: {exc}")


# Live API metrics.
live_quality = calculate_live_quality(df)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Records in this API response",
    f"{len(df):,}",
)

c2.metric(
    "Total studies in registry",
    registry_total,
)

c3.metric(
    "Live data completeness",
    f"{live_quality:.2f}/100",
)

c4.metric(
    "Statuses represented",
    f"{df['Status'].nunique():,}",
)

c5.metric(
    "Sponsors represented",
    f"{df['Sponsor'].nunique():,}",
)

st.caption(
    "Live data completeness is calculated from 12 key fields. "
    "It is not model accuracy or the pipeline validation score."
)


# Live status and phase charts.
left, right = st.columns(2)

with left:
    status_counts = (
        df["Status"]
        .replace("", "Unknown")
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Count")
    )

    st.plotly_chart(
        px.bar(
            status_counts,
            x="Status",
            y="Count",
            title="Statuses in the latest API response",
        ),
        use_container_width=True,
    )

with right:
    phase_counts = (
        df["Phase"]
        .replace("", "Not specified")
        .fillna("Not specified")
        .value_counts()
        .head(12)
        .rename_axis("Phase")
        .reset_index(name="Count")
    )

    st.plotly_chart(
        px.pie(
            phase_counts,
            names="Phase",
            values="Count",
            title="Phases in the latest API response",
        ),
        use_container_width=True,
    )


# Top sponsors chart.
sponsors = (
    df["Sponsor"]
    .replace("", "Unknown")
    .fillna("Unknown")
    .value_counts()
    .head(10)
    .rename_axis("Sponsor")
    .reset_index(name="Count")
)

st.plotly_chart(
    px.bar(
        sponsors,
        x="Count",
        y="Sponsor",
        orientation="h",
        title="Top sponsors in the latest API response",
    ),
    use_container_width=True,
)


# Load existing pipeline validation files.
st.divider()
st.header("Pipeline Validation Summary")

st.caption(
    "The following section reads the saved pipeline dataset and "
    "validation report. It does not run the pipeline."
)

master_df, issue_df, master_error, issues_error = (
    load_pipeline_validation()
)

validation_df = build_validation_table(master_df, issue_df)

if master_error:
    st.warning(
        f"Pipeline master dataset unavailable: {master_error}"
    )

if issues_error:
    st.info(
        f"Validation issues report unavailable: {issues_error}"
    )

if validation_df.empty:
    st.warning(
        "Validation results cannot be displayed. Check that "
        "clinical_trials_master.csv exists and contains a study ID. "
        "Run your pipeline first, then refresh this dashboard."
    )

else:
    total_pipeline = len(validation_df)

    status_values = (
        validation_df["Validation status"]
        .fillna("Unknown")
        .astype(str)
        .str.upper()
    )

    pass_count = int(status_values.eq("PASS").sum())
    review_count = int(
        status_values.eq("REVIEW_REQUIRED").sum()
    )
    other_count = total_pipeline - pass_count - review_count

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Pipeline records",
        f"{total_pipeline:,}",
    )

    m2.metric(
        "Passed validation",
        f"{pass_count:,}",
        f"{pass_count / total_pipeline * 100:.1f}%"
        if total_pipeline else "0%",
    )

    m3.metric(
        "Require review",
        f"{review_count:,}",
        f"{review_count / total_pipeline * 100:.1f}%"
        if total_pipeline else "0%",
    )

    m4.metric(
        "Other / unknown status",
        f"{other_count:,}",
    )

    st.caption(
        "Quality status and validation status are different. "
        "A high quality score does not necessarily mean a study "
        "has passed every validation rule."
    )

    # Validation status chart.
    validation_counts = (
        validation_df["Validation status"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("Validation status")
        .reset_index(name="Count")
    )

    st.plotly_chart(
        px.bar(
            validation_counts,
            x="Validation status",
            y="Count",
            title="Pipeline validation status distribution",
        ),
        use_container_width=True,
    )

    st.subheader("Study Validation Details")

    # Filters for the saved pipeline dataset.
    f1, f2, f3 = st.columns(3)

    validation_options = sorted(
        validation_df["Validation status"]
        .fillna("Unknown")
        .astype(str)
        .unique()
    )

    quality_options = sorted(
        validation_df["Quality status"]
        .fillna("Unknown")
        .astype(str)
        .unique()
    )

    with f1:
        selected_status = st.multiselect(
            "Filter by validation status",
            options=validation_options,
            default=validation_options,
        )

    with f2:
        selected_quality = st.multiselect(
            "Filter by quality status",
            options=quality_options,
            default=quality_options,
        )

    with f3:
        issue_only = st.checkbox(
            "Show only studies with issue details",
            value=False,
        )

    search_text = st.text_input(
        "Search by NCT ID or study title",
        value="",
        placeholder="Example: NCT12345678",
    ).strip().lower()

    filtered = validation_df[
        validation_df["Validation status"]
        .astype(str)
        .isin(selected_status)
        & validation_df["Quality status"]
        .astype(str)
        .isin(selected_quality)
    ].copy()

    if issue_only:
        filtered = filtered[
            filtered["Issue descriptions"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ]

    if search_text:
        matches_id = (
            filtered["NCT ID"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(search_text, na=False)
        )

        matches_title = (
            filtered["Title"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(search_text, na=False)
        )

        filtered = filtered[matches_id | matches_title]

    st.caption(
        f"Displaying {len(filtered):,} of "
        f"{len(validation_df):,} pipeline records."
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quality score": st.column_config.NumberColumn(
                "Quality score",
                format="%.2f",
            ),
        },
    )

    st.download_button(
        "Download filtered validation results as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="clinical_trial_validation_review.csv",
        mime="text/csv",
    )

    # Show the original issue report for detailed inspection.
    if not issue_df.empty:
        with st.expander(
            "View raw pipeline validation issues"
        ):
            raw_issues = issue_df.drop(
                columns=["_study_key"],
                errors="ignore",
            )

            st.dataframe(
                raw_issues,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download complete validation issues CSV",
                data=raw_issues.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="pipeline_validation_issues.csv",
                mime="text/csv",
                key="download_pipeline_issues",
            )


# Latest live records from the API.
st.divider()
st.subheader(
    "Latest Records Fetched Directly from ClinicalTrials.gov"
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Study URL": st.column_config.LinkColumn(
            "Study URL",
            display_text="Open study",
        ),
    },
)

st.download_button(
    "Download live API response as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=(
        f"clinicaltrials_live_"
        f"{fetched_at.strftime('%Y%m%d_%H%M%S')}.csv"
    ),
    mime="text/csv",
    key="download_live_api",
)

st.caption(
    "This dashboard retrieves published clinical-trial registry "
    "metadata, not continuously changing patient data. "
    "Records may remain unchanged when the registry has not "
    "published updates. The live snapshot and pipeline reports "
    "are stored separately."
)



# --------------------------------------------------
# VALIDATION ISSUE ANALYSIS
# --------------------------------------------------

st.divider()
st.header("Validation Issue Analysis")

if issue_df.empty:
    st.info(
        "No validation issue records are available. "
        "Check reports/pipeline_validation_issues.csv."
    )

else:
    issue_type_col = find_column(
        issue_df,
        [
            "issue_type",
            "issue",
            "rule",
            "validation_issue",
            "error_type",
            "issue_description",
            "description",
            "message",
        ],
    )

    issue_id_col = find_column(
        issue_df,
        ["nct_id", "study_id", "NCT ID", "nctid", "studyid"],
    )

    if issue_type_col is None:
        st.warning(
            "The issue report does not contain a recognizable "
            "issue-type column. Available columns: "
            + ", ".join(
                str(col)
                for col in issue_df.columns
                if col != "_study_key"
            )
        )

    else:
        analysis = issue_df.copy()

        analysis["Issue type"] = (
            analysis[issue_type_col]
            .fillna("Unspecified issue")
            .astype(str)
            .str.strip()
            .replace("", "Unspecified issue")
        )

        # Count issue occurrences, not just unique studies.
        summary = (
            analysis.groupby("Issue type")
            .size()
            .reset_index(name="Issue occurrences")
            .sort_values(
                "Issue occurrences",
                ascending=False,
            )
        )

        if issue_id_col:
            unique_studies = (
                analysis.groupby("Issue type")[issue_id_col]
                .nunique()
                .reset_index(
                    name="Unique affected studies"
                )
            )

            summary = summary.merge(
                unique_studies,
                on="Issue type",
                how="left",
            )

        st.caption(
            "Counts are calculated from the saved pipeline "
            "validation issue report. One study may have multiple "
            "issues, so issue occurrences can exceed unique studies."
        )

        st.plotly_chart(
            px.bar(
                summary,
                x="Issue occurrences",
                y="Issue type",
                orientation="h",
                title="Validation Issues by Type",
                text="Issue occurrences",
            ).update_layout(
                yaxis={"categoryorder": "total ascending"}
            ),
            use_container_width=True,
        )

        st.subheader("Issue Summary")
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download issue analysis CSV",
            data=summary.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="validation_issue_analysis.csv",
            mime="text/csv",
            key="download_issue_analysis",
        )

        st.subheader("Studies Affected by Validation Issues")

        selected_issue = st.selectbox(
            "Choose an issue type",
            options=summary["Issue type"].tolist(),
        )

        selected_rows = analysis[
            analysis["Issue type"] == selected_issue
        ].copy()

        display_columns = [
            col for col in selected_rows.columns
            if col != "_study_key"
        ]

        st.dataframe(
            selected_rows[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download affected studies CSV",
            data=selected_rows[display_columns]
            .to_csv(index=False)
            .encode("utf-8"),
            file_name="affected_clinical_studies.csv",
            mime="text/csv",
            key="download_affected_studies",
        )





# --------------------------------------------------
# DATA QUALITY IMPROVEMENT AND CLEANING IMPACT
# --------------------------------------------------

st.divider()
st.header("Data Quality Improvement Analysis")

st.caption(
    "This section displays the actual cleaning operations and "
    "missing-value changes recorded by the data pipeline."
)

CLEANING_SUMMARY_PATH = (
    ROOT / "reports" / "cleaning_impact_summary.csv"
)

MISSING_COMPARISON_PATH = (
    ROOT / "reports" / "missing_values_before_after.csv"
)

if not CLEANING_SUMMARY_PATH.exists():
    st.warning(
        "Cleaning summary not found. Run the pipeline first "
        "to generate reports/cleaning_impact_summary.csv."
    )

elif not MISSING_COMPARISON_PATH.exists():
    st.warning(
        "Missing-value report not found. Run the pipeline first "
        "to generate reports/missing_values_before_after.csv."
    )

else:
    try:
        summary_df = pd.read_csv(
            CLEANING_SUMMARY_PATH,
            keep_default_na=False,
            na_values=[""],
        )

        missing_df = pd.read_csv(
            MISSING_COMPARISON_PATH,
            keep_default_na=False,
            na_values=[""],
        )

        # Convert the saved cleaning metrics into a dictionary.
        metrics = {}

        if {"metric", "value"}.issubset(summary_df.columns):
            for _, row in summary_df.iterrows():
                metrics[str(row["metric"])] = pd.to_numeric(
                    row["value"],
                    errors="coerce",
                )

        def metric_value(name, default=0):
            value = metrics.get(name, default)
            return default if pd.isna(value) else int(value)

        records_before = metric_value("records_before_cleaning")
        records_after = metric_value("records_after_cleaning")
        duplicates_removed = metric_value(
            "records_removed_as_duplicate_ids"
        )
        text_standardized = metric_value(
            "text_values_standardized"
        )
        invalid_enrollment = metric_value(
            "invalid_enrollment_values_converted_to_missing"
        )
        invalid_age = metric_value(
            "invalid_age_values_converted_to_missing"
        )
        invalid_dates = metric_value(
            "invalid_date_values_converted_to_missing"
        )
        columns_added = metric_value("expected_columns_added")

        # Pipeline processing metrics.
        a, b, c = st.columns(3)

        a.metric("Records before cleaning", f"{records_before:,}")
        b.metric("Records after cleaning", f"{records_after:,}")
        c.metric("Duplicate IDs removed", f"{duplicates_removed:,}")

        d, e, f = st.columns(3)

        d.metric("Text values standardized", f"{text_standardized:,}")
        e.metric("Invalid age conversions", f"{invalid_age:,}")
        f.metric(
            "Invalid enrollment conversions",
            f"{invalid_enrollment:,}",
        )

        g, h = st.columns(2)

        g.metric("Invalid date conversions", f"{invalid_dates:,}")
        h.metric("Expected columns added", f"{columns_added:,}")

        st.caption(
            "These counts come from the pipeline's saved cleaning "
            "summary. Invalid conversions are changed to missing "
            "values; they are not automatically corrected."
        )

        # Missing values before and after cleaning.
        required_columns = {
            "column",
            "missing_before_cleaning",
            "missing_after_cleaning",
        }

        if required_columns.issubset(missing_df.columns):
            for col in [
                "missing_before_cleaning",
                "missing_after_cleaning",
            ]:
                missing_df[col] = pd.to_numeric(
                    missing_df[col],
                    errors="coerce",
                ).fillna(0)

            missing_df["Change in missing values"] = (
                missing_df["missing_after_cleaning"]
                - missing_df["missing_before_cleaning"]
            )

            st.subheader("Missing Values Before and After Cleaning")

            chart_df = missing_df.melt(
                id_vars=["column"],
                value_vars=[
                    "missing_before_cleaning",
                    "missing_after_cleaning",
                ],
                var_name="Stage",
                value_name="Missing values",
            )

            chart_df["Stage"] = chart_df["Stage"].map({
                "missing_before_cleaning": "Before cleaning",
                "missing_after_cleaning": "After cleaning",
            })

            st.plotly_chart(
                px.bar(
                    chart_df,
                    x="column",
                    y="Missing values",
                    color="Stage",
                    barmode="group",
                    title="Missing Values by Field",
                ),
                use_container_width=True,
            )

            display_df = missing_df.rename(columns={
                "column": "Column",
                "missing_before_cleaning": "Missing before cleaning",
                "missing_after_cleaning": "Missing after cleaning",
            })

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download cleaning comparison CSV",
                data=display_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="missing_values_before_after.csv",
                mime="text/csv",
                key="download_cleaning_comparison",
            )

        else:
            st.warning(
                "The missing-value report does not contain the "
                "expected columns. Rerun the pipeline to regenerate it."
            )

        # Show the full cleaning summary for transparency.
        with st.expander("View complete pipeline cleaning summary"):
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:
        st.error(
            f"Could not load pipeline cleaning reports: {exc}"
        )


