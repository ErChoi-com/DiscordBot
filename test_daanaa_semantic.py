import sys
sys.path.insert(0, 'src')

from services.job_service import (
    matches_search_parameters_semantic,
    semantic_similarity_score,
    search_text_for_semantic_match,
    normalize_description_text,
    matches_role_filters,
)

daanaa_hw = {
    'title': 'Hardware Electronics Engineer Intern',
    'company': 'Daanaa Resolution',
    'location': 'Vancouver, BC, CA',
    'site_label': 'Indeed',
    'description': 'Daanaa Resolution is seeking a Hardware Electronics Engineer Intern. You will work on embedded hardware design, PCB layout, circuit analysis, electrical testing, and prototype development. Requirements: knowledge of electronics, circuit design, and engineering principles.',
}

daanaa_fw = {
    'title': 'Firmware Engineer Intern',
    'company': 'Daanaa Resolution',
    'location': 'Vancouver, BC, CA',
    'site_label': 'Indeed',
    'description': 'Daanaa Resolution is seeking a Firmware Engineer Intern. You will work on embedded firmware development, microcontroller programming, and hardware-software integration.',
}

daanaa_glassdoor = {
    'title': 'Hardware Electronics Engineer Intern',
    'company': 'Daanaa Resolution',
    'location': 'Vancouver, BC',
    'site_label': 'Glassdoor',
    'description': None,
}

ece_keywords = 'computer software electrical intern'
ece_role_filters = ['internship']
ece_threshold = 0.42

allahu_keywords = 'Electrical engineering intern'
allahu_role_filters = ['internship']
allahu_threshold = 0.30

for job in [daanaa_hw, daanaa_fw, daanaa_glassdoor]:
    title = job['title']
    print("=== " + title + " @ " + job['company'] + " [" + job['site_label'] + "] ===")

    ece_search = search_text_for_semantic_match(ece_keywords, 'Canada', ece_role_filters)
    allahu_search = search_text_for_semantic_match(allahu_keywords, 'Canada', allahu_role_filters)

    desc = normalize_description_text(job.get('description') or '')
    has_desc = bool(desc)

    ece_score = semantic_similarity_score(desc if has_desc else title, ece_search)
    allahu_score = semantic_similarity_score(desc if has_desc else title, allahu_search)

    role_ok = matches_role_filters(title, ece_role_filters)

    print("  role_filters pass: " + str(role_ok))
    print("  has description:   " + str(has_desc))
    print("  ECE search:    " + ece_search)
    print("  Allahu search: " + allahu_search)
    print("  Score vs ECE (thr 0.42):    " + str(round(ece_score, 4)) + " -> " + ("PASS" if ece_score >= ece_threshold else "FILTERED OUT"))
    print("  Score vs allahu (thr 0.30): " + str(round(allahu_score, 4)) + " -> " + ("PASS" if allahu_score >= allahu_threshold else "FILTERED OUT"))

    # Also test actual matches_search_parameters_semantic
    ece_match = matches_search_parameters_semantic(job, ece_keywords, 'Canada', ece_role_filters, threshold=ece_threshold)
    allahu_match = matches_search_parameters_semantic(job, allahu_keywords, 'Canada', allahu_role_filters, threshold=allahu_threshold)
    print("  matches_semantic ECE:    " + str(ece_match))
    print("  matches_semantic allahu: " + str(allahu_match))
    print()
