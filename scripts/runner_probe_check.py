"""Read-only check of whether a GitHub runner can validate ATS boards.

Answers one empirical question and writes nothing at all: no dead marks, no
confirmed-live marks, no files. It probes a fixed control set of boards that
were confirmed live from the bot's own host, and reports what a runner is told
about the same companies.

The point is that both sides of the comparison are known. If a runner gets 200
for boards this machine also sees as live, the platform does not care where the
request came from. If it gets 403, 429 or a timeout for boards that are
definitely alive, the block is real and validation must stay off the runners --
because validate_ats_slugs writes exactly that answer into dead_slugs, and the
bot reads those files.

This exists because the standing decision (2026-08-24) rests on a measurable
claim, and a measurable claim should be measured rather than remembered.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_ats_slugs as v  # noqa: E402

CONTROL = {
    "greenhouse": [
        "sitelineinc",
        "rvohcontentfreelance",
        "texaschillersystems",
        "swandermatology",
        "wurljobs",
        "candidaturasdirecionadasxpinc",
        "egineering",
        "rightwayhealthcare"
    ],
    "lever": [
        "cherre",
        "metr",
        "wachter",
        "engelvoelkers",
        "americaatwork",
        "projectkittyhawk",
        "premiertruck",
        "boxbot"
    ],
    "ashby": [
        "beyondsports",
        "brightwheel",
        "gadget",
        "lpadesignstudios",
        "zoe",
        "onoshealth",
        "laurel",
        "jbs-dev"
    ],
    "workday": [
        "crisprtx|wd12|careers",
        "cookchildrens|wd1|cook_childrens_careers",
        "ferguson|wd1|ferguson_campus",
        "pensacolastate|wd501|adjunct_faculty_site",
        "roche|wd3|roche-ext",
        "fox|wd1|foxtvst_east",
        "flir|wd1|flircareers",
        "ikusi|wd3|vacantes_sitio_externo"
    ],
    "icims": [
        "calamp",
        "viapath",
        "spanishcareers-grimmway",
        "secure-energy",
        "nyfoundling",
        "brasfieldgorrie",
        "exponent",
        "tmacrestaurants"
    ],
    "bamboohr": [
        "elementalenzymes",
        "allsaintsdayschool",
        "cybera",
        "bioratherapeutics",
        "bekhealthcorp",
        "crchc",
        "envirochemservices",
        "spc1"
    ],
    "paylocity": [
        "7d69df87-2579-4659-b811-39f3c0fa4cda",
        "883eadf0-53a6-47d9-977b-b37057f40679",
        "71402a42-2fb0-421d-9947-fc1b1bc54339",
        "483822c7-dd52-43bc-b0b7-6dbd7db7b96b",
        "33abf79f-46e4-485d-8654-a107863253b1",
        "0a09b482-a527-4e77-b98d-065574bb01be",
        "7581471c-46bd-48af-9074-381190baab55",
        "6772058e-f2dd-40f7-9e01-225abcb22d48"
    ],
    "workable": [
        "al-jomaih-energy-and-water",
        "aldrin",
        "aardvark-studios",
        "1871",
        "amazing-athletes-2",
        "alphax",
        "adree",
        "animal-dynamics"
    ],
    "breezy": [
        "seeknow",
        "twacareer",
        "mozaic",
        "clove-twine",
        "sparkle-freshness",
        "framework",
        "real-estate-webmasters",
        "mezzanine-ware"
    ],
    "smartrecruiters": [
        "captechconsulting",
        "cdprojektred",
        "intellihub1",
        "sottassa",
        "zenosynekft",
        "bhft",
        "smartkarma",
        "blackbirdcollective"
    ],
    "rippling": [
        "the-samaritans-cape-cod-and-the-islands",
        "pet-screening-job-board",
        "kitchen-food-company-ltd",
        "green-impact-exchange",
        "d-wave-quantum",
        "botbuilt",
        "purefacts_jobs",
        "anthologic"
    ],
    "teamtailor": [
        "metanet",
        "creativelivesinprogress",
        "jochenschweizermydaysholdinggmbh-1734018413",
        "clickboat",
        "vaccindirektisverigeab",
        "autorolaaustria",
        "billdu",
        "tenpo"
    ],
    "jazzhr": [
        "kscourts",
        "360careers",
        "ltclanguagesolutions",
        "petersontechnologies",
        "satellitesheltersinc",
        "tooli",
        "beasleymediagroup",
        "ecapital"
    ],
    "recruitee": [
        "bundl",
        "novutech",
        "smartrobotics",
        "jobsatlanticvcfoodlabs",
        "60secondstonapoli",
        "baeckereisipl",
        "surfer",
        "bsi"
    ],
    "jobvite": [
        "buckman-fr",
        "mavenlink",
        "imagine-learning",
        "bowl",
        "leantechio",
        "ardelyx",
        "altamiracorps",
        "fashionphilecareers"
    ],
    "applicantpro": [
        "hhbldrs",
        "zund",
        "aiaa",
        "elitekingwood",
        "apcosigns",
        "dbaconstruction",
        "nationalwrecker",
        "ctgreenbank"
    ]
}


def main() -> int:
    verdicts: dict[str, dict] = {}
    for platform, slugs in CONTROL.items():
        probe = v.PROBES.get(platform)
        if probe is None:
            continue
        counts: collections.Counter = collections.Counter()
        for slug in slugs:
            try:
                counts["live" if probe(slug) else "dead"] += 1
            except v.Unreachable as exc:
                counts[f"unreachable:{exc}"] += 1
            except Exception as exc:  # noqa: BLE001
                counts[f"error:{type(exc).__name__}"] += 1
        live = counts.get("live", 0)
        verdicts[platform] = {"n": len(slugs), "live": live,
                              "detail": dict(counts)}
        print(f"[probe] {platform:16} {live}/{len(slugs)} still read as live  {dict(counts)}")

    agree = sum(x["live"] for x in verdicts.values())
    total = sum(x["n"] for x in verdicts.values())
    print(f"[probe] agreement with the local verdict: {agree}/{total}")
    print(json.dumps({"agreement": [agree, total], "platforms": verdicts}, indent=2))
    # Always exits 0. This reports; it does not gate anything.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
