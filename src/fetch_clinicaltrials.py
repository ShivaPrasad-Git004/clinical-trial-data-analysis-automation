
"""Fetch and flatten clinical-trial data from ClinicalTrials.gov API v2."""

import time

import pandas as pd
import requests

API_URL = "https://clinicaltrials.gov/api/v2/studies"
MAX_RECORDS = 5000
MAX_RETRIES = 3

FIELDS = [
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "Condition",
    "LeadSponsorName",
    "LeadSponsorClass",
    "StudyType",
    "Phase",
    "StartDate",
    "CompletionDate",
    "LastUpdatePostDate",
    "EnrollmentCount",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "InterventionName",
    "InterventionType",
    "LocationCountry",
    "DesignAllocation",
    "DesignInterventionModel",
    "DesignMasking",
    "PrimaryOutcomeMeasure",
]


def get_nested(data, *keys, default=None):
    """Safely retrieve a value from nested dictionaries."""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key)

    return data if data is not None else default


def join_values(values):
    """Join available values into one string."""
    if not isinstance(values, (list, tuple)):
        return "" if values is None else str(values)

    return "; ".join(
        str(value)
        for value in values
        if value is not None and str(value).strip()
    )


def extract_study(study):
    """Convert one API study into a flat dictionary."""
    protocol = study.get("protocolSection", {})

    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    design = protocol.get("designModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    conditions = protocol.get("conditionsModule", {})
    interventions_module = protocol.get("armsInterventionsModule", {})
    locations_module = protocol.get("contactsLocationsModule", {})
    outcomes = protocol.get("outcomesModule", {})

    design_info = design.get("designInfo", {})
    enrollment_info = design.get("enrollmentInfo", {})
    lead_sponsor = sponsor.get("leadSponsor", {}) or {}

    interventions = interventions_module.get("interventions", []) or []
    locations = locations_module.get("locations", []) or []
    primary_outcomes = outcomes.get("primaryOutcomes", []) or []

    return {
        "nct_id": identification.get("nctId"),
        "brief_title": identification.get("briefTitle"),
        "overall_status": status.get("overallStatus"),
        "condition": join_values(conditions.get("conditions", [])),
        "sponsor": lead_sponsor.get("name"),
        "agency_class": lead_sponsor.get("class"),
        "study_type": design.get("studyType"),
        "phase": join_values(design.get("phases", [])),
        "enrollment": enrollment_info.get("count"),
        "start_date": get_nested(
            status, "startDateStruct", "date"
        ),
        "completion_date": get_nested(
            status, "completionDateStruct", "date"
        ),
        "last_update_post_date": get_nested(
            status, "lastUpdatePostDateStruct", "date"
        ),
        "minimum_age": eligibility.get("minimumAge"),
        "maximum_age": eligibility.get("maximumAge"),
        "sex": eligibility.get("sex"),
        "intervention": join_values(
            [item.get("name") for item in interventions]
        ),
        "intervention_type": join_values(
            [item.get("type") for item in interventions]
        ),
        "country": join_values(
            [item.get("country") for item in locations]
        ),
        "allocation": design_info.get("allocation"),
        "intervention_model": design_info.get("interventionModel"),
        "masking": get_nested(
            design_info, "maskingInfo", "masking"
        ),
        "primary_outcome": join_values(
            [item.get("measure") for item in primary_outcomes]
        ),
    }


def fetch_clinical_trials(
    page_size=1000,
    max_records=MAX_RECORDS,
):
    """Fetch at most max_records studies from ClinicalTrials.gov."""
    if max_records < 1:
        raise ValueError("max_records must be at least 1")

    page_size = min(max(1, page_size), 1000)

    all_rows = []
    page_token = None
    total_count = 0

    with requests.Session() as session:
        while len(all_rows) < max_records:
            remaining = max_records - len(all_rows)
            current_page_size = min(page_size, remaining)

            params = {
                "format": "json",
                "pageSize": current_page_size,
                "countTotal": "true",
                "sort": "LastUpdatePostDate:desc",
                "fields": ",".join(FIELDS),
            }

            if page_token:
                params["pageToken"] = page_token

            response = None

            for attempt in range(MAX_RETRIES):
                try:
                    response = session.get(
                        API_URL,
                        params=params,
                        timeout=120,
                    )

                    if (
                        response.status_code == 429
                        or response.status_code >= 500
                    ):
                        if attempt < MAX_RETRIES - 1:
                            retry_after = response.headers.get(
                                "Retry-After"
                            )
                            try:
                                delay = (
                                    min(float(retry_after), 60)
                                    if retry_after
                                    else 2 ** attempt
                                )
                            except (ValueError, TypeError):
                                delay = 2 ** attempt

                            time.sleep(delay)
                            continue

                    response.raise_for_status()
                    break

                except requests.RequestException:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    time.sleep(2 ** attempt)

            if response is None:
                raise RuntimeError("API request did not return a response")

            data = response.json()
            total_count = data.get("totalCount", total_count)
            studies = data.get("studies", [])

            if not studies:
                break

            all_rows.extend(
                extract_study(study)
                for study in studies[:remaining]
            )

            print(
                f"Fetched {len(all_rows):,} "
                f"of maximum {max_records:,} records"
            )

            if len(all_rows) >= max_records:
                break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    df = pd.DataFrame(all_rows)

    print(f"Records fetched: {len(df):,}")
    print(f"Total studies reported by API: {total_count:,}")

    if "agency_class" in df.columns:
        print("\nSponsor classification counts:")
        print(df["agency_class"].value_counts(dropna=False).head(10))

    return df, total_count
