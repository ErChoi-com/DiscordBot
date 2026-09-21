from services.jba import semantic_dedup


def test_normalize_title_for_dedup():
    assert semantic_dedup.normalize_title_for_dedup("Retail Sales Associate - Part Time") == "retail sales associate part time"
    assert semantic_dedup.normalize_title_for_dedup("Senior Backend Engineer (Remote)") == "senior backend engineer remote"
    assert semantic_dedup.normalize_title_for_dedup("Commercial Driver -- Full-Time") == "commercial driver full time"


def test_find_semantic_duplicates_respects_title_guards():
    jobs = [
        {"company": "Chili's", "title": "Server", "location": "Cleburne", "description": "Serve food in restaurant."},
        {"company": "Chili's", "title": "Host", "location": "Cleburne", "description": "Serve food in restaurant."},
        {"company": "Oracle", "title": "Software Engineer", "location": "Austin", "description": "Build cloud services."},
        {"company": "Oracle", "title": "Software Engineer", "location": "Austin", "description": "Build cloud services with different URL."},
    ]
    dups = semantic_dedup.find_semantic_duplicates(jobs, threshold=0.85)
    # Server and Host must NOT be duplicates despite identical description
    for idx1, idx2, score in dups:
        assert not (jobs[idx1]["title"] == "Server" and jobs[idx2]["title"] == "Host")

    # The two Software Engineer jobs SHOULD be recognized as duplicates
    se_dups = [
        (jobs[i]["title"], jobs[j]["title"])
        for i, j, score in dups
        if "Software Engineer" in jobs[i]["title"] and "Software Engineer" in jobs[j]["title"]
    ]
    assert len(se_dups) >= 1
