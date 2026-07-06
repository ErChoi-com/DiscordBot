# geolocation.py
"""
Reusable geolocation module.
Parses job posting location strings and looks up coordinates.

Usage:
    from geolocation import build_lookup, lookup_location

    maps = build_lookup("data/locations.json")
    result = lookup_location("San Francisco, CA", maps)
    # result => {"remote": False, "coords": [37.7749, -122.4194]}
"""

import json
import re
import unicodedata
from pathlib import Path


# ---- Constants ----

REMOTE_KEYWORDS = {
    "remote",
    "anywhere",
    "worldwide",
    "work from home",
    "wfh",
}

TIMEZONE_KEYWORDS = {
    "time zone",
    "timezone",
    "time zones",
    "timezones",
}

GARBAGE_LOCATIONS = {
    "",
    "not specified",
    "n/a",
    "none",
    "tbd",
    "unspecified",
    "multiple locations",
    "various",
    "flexible",
    "other",
    "global",
    "multiple",
    "varies",
    "various locations",
    "2 locations",
    "3 locations",
    "4 locations",
    "5 locations",
    "6 locations",
    "7 locations",
    "8 locations",
    "9 locations",
    "10 locations",
}

WORK_ARRANGEMENT_PREFIXES = [
    "hybrid in ",
    "hybrid - ",
    "hybrid: ",
    "hybrid, ",
    "on-site in ",
    "on site in ",
    "onsite in ",
    "in-office in ",
    "in office in ",
    "based in ",
    "located in ",
]

DIRECTION_EXPANSIONS = {
    " n ": " north ",
    " s ": " south ",
    " e ": " east ",
    " w ": " west ",
    " nw ": " northwest ",
    " ne ": " northeast ",
    " sw ": " southwest ",
    " se ": " southeast ",
}

ABBREVIATION_EXPANSIONS = {
    " ft ": " fort ",
    " mt ": " mount ",
    " pt ": " port ",
}

COUNTRY_ALIASES = {
    # North America
    "us": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US",
    "united states": "US", "united states of america": "US", "america": "US",
    # "ca": "CA",  # collides with California - use "can"/"canada" instead
    "can": "CA", "canada": "CA",
    "mx": "MX", "mex": "MX", "mexico": "MX",
    # UK / Ireland
    "gb": "GB", "gbr": "GB", "uk": "GB", "u.k.": "GB",
    "united kingdom": "GB", "england": "GB", "scotland": "GB",
    "wales": "GB", "northern ireland": "GB", "britain": "GB", "great britain": "GB",
    "ie": "IE", "irl": "IE", "ireland": "IE",
    # Europe
    # "de": "DE",  # collides with Delaware
    "deu": "DE", "ger": "DE", "germany": "DE", "deutschland": "DE",
    "fr": "FR", "fra": "FR", "france": "FR",
    "es": "ES", "esp": "ES", "spain": "ES",
    "it": "IT", "ita": "IT", "italy": "IT",
    "nl": "NL", "nld": "NL", "netherlands": "NL", "holland": "NL",
    "be": "BE", "bel": "BE", "belgium": "BE",
    "ch": "CH", "che": "CH", "switzerland": "CH",
    "at": "AT", "aut": "AT", "austria": "AT",
    "se": "SE", "swe": "SE", "sweden": "SE",
    "no": "NO", "nor": "NO", "norway": "NO",
    "dk": "DK", "dnk": "DK", "denmark": "DK",
    "fi": "FI", "fin": "FI", "finland": "FI",
    "pl": "PL", "pol": "PL", "poland": "PL",
    "pt": "PT", "prt": "PT", "portugal": "PT",
    "cz": "CZ", "cze": "CZ", "czech republic": "CZ", "czechia": "CZ",
    "gr": "GR", "grc": "GR", "greece": "GR",
    "ro": "RO", "rou": "RO", "romania": "RO",
    "ua": "UA", "ukr": "UA", "ukraine": "UA",
    # Asia / Pacific
    # "in": "IN",  # collides with Indiana
    "ind": "IN", "india": "IN",
    "cn": "CN", "chn": "CN", "china": "CN",
    "jp": "JP", "jpn": "JP", "japan": "JP",
    "kr": "KR", "kor": "KR", "korea": "KR", "south korea": "KR",
    "sg": "SG", "sgp": "SG", "singapore": "SG",
    "my": "MY", "mys": "MY", "malaysia": "MY",
    "ph": "PH", "phl": "PH", "philippines": "PH",
    # "id": "ID",  # collides with Idaho
    "idn": "ID", "indonesia": "ID",
    "th": "TH", "tha": "TH", "thailand": "TH",
    "vn": "VN", "vnm": "VN", "vietnam": "VN",
    "hk": "HK", "hkg": "HK", "hong kong": "HK",
    "tw": "TW", "twn": "TW", "taiwan": "TW",
    "au": "AU", "aus": "AU", "australia": "AU",
    "nz": "NZ", "nzl": "NZ", "new zealand": "NZ",
    # Middle East / Africa
    # "il": "IL",  # collides with Illinois
    "isr": "IL", "israel": "IL",
    "ae": "AE", "are": "AE", "united arab emirates": "AE", "uae": "AE",
    # "sa": "SA",  # SA isn't a US state but keeping lowercase-only reduces risk
    "sau": "SA", "saudi arabia": "SA",
    "tr": "TR", "tur": "TR", "turkey": "TR",
    "za": "ZA", "zaf": "ZA", "south africa": "ZA",
    "eg": "EG", "egy": "EG", "egypt": "EG",
    "ng": "NG", "nga": "NG", "nigeria": "NG",
    "ke": "KE", "ken": "KE", "kenya": "KE",
    # South America
    "br": "BR", "bra": "BR", "brazil": "BR", "brasil": "BR",
    "ar": "AR", "arg": "AR", "argentina": "AR",
    "cl": "CL", "chl": "CL", "chile": "CL",
    # "co": "CO",  # collides with Colorado
    "col": "CO", "colombia": "CO",
    "pe": "PE", "per": "PE", "peru": "PE",
}

def _expand_code(code: str) -> str:
    """Return what normalize() does to a bare 2-letter code (e.g. 'NE' → 'northeast')."""
    padded = f" {code.lower()} "
    for abbrev, full in {**DIRECTION_EXPANSIONS, **ABBREVIATION_EXPANSIONS}.items():
        padded = padded.replace(abbrev, full)
    return padded.strip()


def _build_us_states() -> dict[str, str]:
    result = {}
    try:
        import pycountry
        for sub in pycountry.subdivisions.get(country_code="US"):
            code = sub.code.split("-")[1]
            raw = code.lower()
            result[raw] = code
            result[sub.name.lower()] = code
            # normalize() expands MT→"mount" and NE→"northeast"; add the expanded
            # form so "Billings, MT" still resolves after the full string is normalized
            expanded = _expand_code(code)
            if expanded != raw:
                result[expanded] = code
    except ImportError:
        pass
    result.update({"d c": "DC", "washington dc": "DC", "washington d c": "DC"})
    return result


def _build_ca_provinces() -> dict[str, str]:
    result = {}
    try:
        import pycountry
        for sub in pycountry.subdivisions.get(country_code="CA"):
            code = sub.code.split("-")[1].lower()
            name = sub.name.lower()
            result[code] = name
            result[name] = name
    except ImportError:
        pass
    result.update({"newfoundland": "newfoundland and labrador", "pei": "prince edward island"})
    return result


US_STATES = _build_us_states()
CA_PROVINCES = _build_ca_provinces()

# City aliases — normalized input → form that matches GeoNames asciiname
# Direction: job posting says X → look it up as Y in locations.json
CITY_ALIASES = {
    # India — GeoNames uses modern official names
    "bangalore": "bengaluru",
    "bombay": "mumbai",
    "madras": "chennai",
    "calcutta": "kolkata",
    # China
    "peking": "beijing",
    # Common US disambiguations / abbreviations
    "new york": "new york city",
    "nyc": "new york city",
    "ny city": "new york city",
    "la": "los angeles",
    "sf": "san francisco",
    "san fran": "san francisco",
    # UK special
    "london city of": "london",
    "city of london": "london",
    # Metro areas → main city
    "bay area": "san francisco",
    "sf bay area": "san francisco",
    "san francisco bay area": "san francisco",
    "greater boston": "boston",
    "boston metro": "boston",
    "nyc metro": "new york city",
    "ny metro": "new york city",
    "greater new york": "new york city",
    "new york metro": "new york city",
    "dc metro": "washington",
    "washington metro": "washington",
    "la metro": "los angeles",
    "greater los angeles": "los angeles",
    "greater chicago": "chicago",
    "chicago metro": "chicago",
    "greater seattle": "seattle",
    "seattle metro": "seattle",
    "greater london": "london",
    "london metro": "london",
}

NYC_BOROUGHS = {
    "bronx", "brooklyn", "queens", "staten island", "manhattan",
}

JUNK_TOKEN_SUFFIXES = {
    "hq",
    "office",
    "headquarters",
    "hub",
    "campus",
    "location",
    "site",
    "center",
    "centre",
    "area",
    "township",
    "twp",
}

FAMOUS_CITY_DEFAULTS = {
    # city_name → country code. Used only when city appears alone.
    "london": "GB",
    "paris": "FR",
    "san francisco": "US",
    "moscow": "RU",
    "berlin": "DE",
    "madrid": "ES",
    "rome": "IT",
    "sydney": "AU",
    "toronto": "CA",
    "dublin": "IE",
    "athens": "GR",
    "vienna": "AT",
    "cairo": "EG",
    "boston": "US",
    "chicago": "US",
    "seattle": "US",
    "denver": "US",
    "portland": "US",  # Portland OR, not Portland ME
    "columbus": "US",
    "richmond": "US",
    "springfield": "US",
}


# ---- Normalization ----


def normalize(s):
    """Lowercase, strip diacritics, strip punctuation, normalize saint/directions/abbrevs, collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    for ch in [".", "'", "`"]:
        s = s.replace(ch, "")
    s = " ".join(s.lower().split())

    if s.startswith("saint "):
        s = "st " + s[6:]
    s = s.replace(" saint ", " st ")

    padded = f" {s} "
    for abbrev, full in DIRECTION_EXPANSIONS.items():
        padded = padded.replace(abbrev, full)
    for abbrev, full in ABBREVIATION_EXPANSIONS.items():
        padded = padded.replace(abbrev, full)
    s = padded.strip()

    return s


# ---- Remote detection ----


def is_remote(location_str):
    """True if the location string indicates a remote job."""
    if not location_str:
        return False
    s = normalize(location_str)
    if s in GARBAGE_LOCATIONS:
        return False
    return any(kw in s for kw in REMOTE_KEYWORDS)


# ---- Country / admin / city extraction ----


def extract_country(tokens):
    if len(tokens) <= 1:
        return None, tokens
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i]
        if t in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[t], tokens[:i] + tokens[i + 1 :]
    return None, tokens


def extract_us_state(tokens):
    if len(tokens) <= 1:
        return None, tokens
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i]
        if t in US_STATES:
            return US_STATES[t], tokens[:i] + tokens[i + 1 :]
    return None, tokens


def extract_ca_province(tokens):
    if len(tokens) <= 1:
        return None, tokens
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i]
        if t in CA_PROVINCES:
            return CA_PROVINCES[t], tokens[:i] + tokens[i + 1 :]
    return None, tokens


def clean_token(t):
    """Strip trailing junk words from a token."""
    words = t.split()
    while words and words[-1] in JUNK_TOKEN_SUFFIXES:
        words.pop()
    return " ".join(words)


def strip_work_arrangement(normalized):
    for prefix in WORK_ARRANGEMENT_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def parse_job_location(location_str):
    result = {"remote": False, "city": None, "admin": None, "country": None}

    if not location_str:
        return result

    normalized = normalize(location_str)

    if not normalized or normalized in GARBAGE_LOCATIONS:
        return result

    if any(kw in normalized for kw in REMOTE_KEYWORDS):
        result["remote"] = True
        return result

    if any(kw in normalized for kw in TIMEZONE_KEYWORDS):
        result["remote"] = True
        return result

    normalized = strip_work_arrangement(normalized)
    
    # Strip parenthetical suffixes: "(HQ)", "(Main Office)", "(Remote)", etc.
    normalized = re.sub(r'\s*\([^)]*\)\s*', ' ', normalized).strip()
    
    for pattern in ["- remote", "— remote"]:
        normalized = normalized.replace(pattern, "")
    normalized = normalized.strip(" ,-—")

    if not normalized:
        return result

    tokens = [clean_token(t.strip()) for t in normalized.split(",") if t.strip()]
    tokens = [t for t in tokens if t]

    if not tokens:
        return result

    # Dedupe consecutive identical tokens
    deduped = []
    for t in tokens:
        if not deduped or deduped[-1] != t:
            deduped.append(t)
    tokens = deduped

    joined = " ".join(tokens)
    if joined in CITY_ALIASES:
        result["city"] = CITY_ALIASES[joined]
        return result

    # Handle "city state" space-separated patterns
    if len(tokens) == 1:
        words = tokens[0].split()
        if len(words) >= 2 and words[-1] in US_STATES:
            state = words[-1]
            city = " ".join(words[:-1])
            tokens = [city, state]
    
    # NEW: Single-token country-only case (e.g. "US", "UK", "France")
    # This must come BEFORE extract_country which bails on single tokens
    if len(tokens) == 1 and tokens[0] in COUNTRY_ALIASES:
        result["country"] = COUNTRY_ALIASES[tokens[0]]
        # No city — just a country. Return with country set, no coords will match.
        return result

    # Extract province/state before country so "NL"/"PE" resolve as
    # Canadian provinces rather than Netherlands/Peru when a city precedes them.
    province, tokens_after_prov = extract_ca_province(tokens)
    if province:
        result["admin"] = province
        tokens = tokens_after_prov
        result["country"] = "CA"
    else:
        state, tokens_after_state = extract_us_state(tokens)
        if state:
            result["admin"] = state
            tokens = tokens_after_state
            result["country"] = "US"

    country, tokens = extract_country(tokens)
    if country:
        result["country"] = country

    if tokens:
        if result["admin"] is None and len(tokens) >= 2:
            last = tokens[-1]
            result["admin"] = last
            result["city"] = " ".join(tokens[:-1])
        elif len(tokens) == 1:
            result["city"] = tokens[0]
        else:
            result["city"] = tokens[0]

    if result["city"] and result["city"] in CITY_ALIASES:
        result["city"] = CITY_ALIASES[result["city"]]

    return result


# ---- Build lookup maps ----


def build_lookup():
    try:
        from geo_db import build_lookup_from_db  # direct import (scraper.py context)
    except ImportError:
        from services.jba.geo_db import build_lookup_from_db  # package import context
    return build_lookup_from_db()


# ---- Main lookup function ----


def lookup_location(location_str, maps):
    parsed = parse_job_location(location_str)

    if parsed["remote"]:
        return {"remote": True, "coords": None}

    city = parsed["city"]
    admin = parsed["admin"].lower() if parsed["admin"] else None
    country = parsed["country"]

    if not city:
        return {"remote": False, "coords": None}
    
    # Map NYC boroughs to New York City (only when admin explicitly NY)
    if city in NYC_BOROUGHS and admin == "ny":
        city = "new york city"

    if city and admin and country:
        coords = maps["city_admin_country"].get(f"{city}|{admin}|{country}")
        if coords:
            return {"remote": False, "coords": coords}

    if city and country:
        coords = maps["city_country"].get(f"{city}|{country}")
        if coords:
            return {"remote": False, "coords": coords}

    if city and admin:
        coords = maps["city_admin"].get(f"{city}|{admin}")
        if coords:
            return {"remote": False, "coords": coords}

    # NEW: famous-city default before city-only fallback
    if city in FAMOUS_CITY_DEFAULTS and not country:
        default_country = FAMOUS_CITY_DEFAULTS[city]
        coords = maps["city_country"].get(f"{city}|{default_country}")
        if coords:
            return {"remote": False, "coords": coords}

    coords = maps["city"].get(city)
    if coords:
        return {"remote": False, "coords": coords}

    return {"remote": False, "coords": None}