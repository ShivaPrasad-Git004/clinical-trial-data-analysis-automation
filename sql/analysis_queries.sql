USE clinical_trial_db;

SELECT overall_status, COUNT(*) AS trial_count FROM clinical_trials GROUP BY overall_status ORDER BY trial_count DESC;
SELECT phase, COUNT(*) AS trial_count FROM clinical_trials GROUP BY phase ORDER BY trial_count DESC;
SELECT sponsor, COUNT(*) AS trial_count FROM clinical_trials WHERE sponsor IS NOT NULL GROUP BY sponsor ORDER BY trial_count DESC LIMIT 10;
SELECT country, COUNT(*) AS trial_count FROM clinical_trials GROUP BY country ORDER BY trial_count DESC;
SELECT phase, ROUND(AVG(enrollment),2) AS avg_enrollment FROM clinical_trials WHERE enrollment > 0 GROUP BY phase ORDER BY avg_enrollment DESC;
SELECT quality_status, COUNT(*) AS records FROM clinical_trials GROUP BY quality_status ORDER BY records DESC;
SELECT anomaly_flag, COUNT(*) AS records FROM clinical_trials GROUP BY anomaly_flag;
SELECT nct_id, minimum_age, maximum_age FROM clinical_trials WHERE minimum_age > maximum_age;
SELECT nct_id, start_date, completion_date FROM clinical_trials WHERE completion_date IS NOT NULL AND completion_date < start_date;
SELECT phase, overall_status, COUNT(*) AS trial_count FROM clinical_trials GROUP BY phase, overall_status ORDER BY phase, trial_count DESC;
