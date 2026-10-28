CREATE DATABASE IF NOT EXISTS clinical_trial_db;
USE clinical_trial_db;

CREATE TABLE IF NOT EXISTS clinical_trials (
    nct_id VARCHAR(20) PRIMARY KEY,
    brief_title TEXT,
    overall_status VARCHAR(50),
    study_type VARCHAR(50),
    phase VARCHAR(30),
    enrollment INT,
    start_date DATE,
    completion_date DATE,
    condition VARCHAR(255),
    intervention VARCHAR(255),
    intervention_type VARCHAR(100),
    sponsor VARCHAR(255),
    agency_class VARCHAR(100),
    country VARCHAR(100),
    minimum_age INT,
    maximum_age INT,
    sex VARCHAR(20),
    allocation VARCHAR(50),
    intervention_model VARCHAR(50),
    masking VARCHAR(50),
    primary_outcome TEXT,
    quality_score DECIMAL(5,2),
    quality_status VARCHAR(30),
    anomaly_flag TINYINT DEFAULT 0,
    anomaly_score DECIMAL(10,6)
);
CREATE INDEX idx_status ON clinical_trials(overall_status);
CREATE INDEX idx_phase ON clinical_trials(phase);
CREATE INDEX idx_sponsor ON clinical_trials(sponsor);
CREATE INDEX idx_country ON clinical_trials(country);
