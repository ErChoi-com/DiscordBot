"""Started as a copy of https://github.com/Feashliaa/job-board-aggregator.

That is no longer what this package is, and the description that said so was
worth correcting: it told the next reader these files are verbatim copies to be
left alone and reached only through ats_service, when six of the seven are
imported directly by the bot and have been rewritten around its needs.

ADOPTED -- owned by this repo, imported directly, edit freely:

    merge_data        the archive writer; 14 importers outside this package
    geo_db            GeoNames lookups; 8
    geolocation       location parsing; 4
    archive_index     the SQLite dedup index behind merge_data
    description_cache descriptions keyed by URL
    geo_priority      which boards post in which country, which decides fleet
                      order in watchers.manager._ordered_slugs

VENDORED -- still an unmodified copy, imported by nothing:

    scraper           the aggregator's own ATS fetchers, superseded by
                      services.ats_service. Its "TODO - Add Workable" is
                      upstream's text, not an open item here; ats_service has
                      scraped Workable for as long as the platform list has
                      existed. Kept as the reference the adopted modules were
                      derived from, so a question about why one of them behaves
                      the way it does has an answer that is not a guess.

tests/test_jba_package_docs.py keeps these two lists honest, because a file
quietly moving from the second group to the first is exactly how the previous
description stopped being true.
"""
