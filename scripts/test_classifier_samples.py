"""Live sample testing for converted Matryoshka classifiers across dimensions."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from services.semantic.engine import get_semantic_engine

engine = get_semantic_engine()
print("Registered Tasks:", engine.get_registered_tasks())

seniority_test_cases = [
    ("Software Engineering Intern (Summer 2026)", "intern"),
    ("New Grad Software Developer 2026", "newgrad"),
    ("Associate QA Automation Engineer", "junior"),
    ("Full Stack Web Developer", "mid"),
    ("Senior Distributed Systems Engineer", "senior"),
    ("Principal Infrastructure Architect", "staff"),
    ("Senior Intern Program Coordinator", "staff"),
    ("Director of Early Career Programs", "staff"),
    ("Staff Machine Learning Engineer - LLM Reasoning", "staff"),
    ("Co-op Software Developer - Cloud Infrastructure", "intern"),
]

print("\n" + "=" * 95)
print("SENIORITY CLASSIFIER: LIVE SAMPLES (384-d Production vs 128-d Compressed)")
print("=" * 95)
header = f"{'Job Title':<48} | {'Exp':<8} | {'384-d Pred (Conf)':<18} | {'128-d Pred (Conf)':<18}"
print(header)
print("-" * 95)

for title, exp in seniority_test_cases:
    pred_384, conf_384 = engine.classify_seniority(title, dim=384)
    pred_128, conf_128 = engine.classify_seniority(title, dim=128)
    ok_384 = "[+]" if pred_384 == exp else "[-]"
    ok_128 = "[+]" if pred_128 == exp else "[-]"
    s_384 = f"{ok_384} {pred_384} ({conf_384*100:.1f}%)"
    s_128 = f"{ok_128} {pred_128} ({conf_128*100:.1f}%)"
    print(f"{title[:47]:<48} | {exp:<8} | {s_384:<18} | {s_128:<18}")

posting_test_cases = [
    (
        "Architect and deploy distributed microservices in Go and Python using Kafka, gRPC, and PostgreSQL.",
        "SKILL_DUTY",
    ),
    (
        "Fine-tune large language models and design reinforcement learning reward functions in PyTorch.",
        "SKILL_DUTY",
    ),
    (
        "Must have a valid Driver's License and ability to lift 50 lbs on the warehouse floor.",
        "SKILL_DUTY",
    ),
    (
        "We are an Equal Opportunity Employer and do not discriminate based on race, religion, gender, or origin.",
        "BOILERPLATE",
    ),
    (
        "We offer comprehensive health, dental, and vision insurance plus a 401(k) retirement match.",
        "BOILERPLATE",
    ),
    (
        "COVID-19 vaccination is mandatory for all on-site personnel per corporate health policy.",
        "BOILERPLATE",
    ),
    (
        "Base salary range: $165,000 - $210,000 USD plus equity and performance bonus.",
        "ROLE_FACTS",
    ),
    (
        "Location: Hybrid, 3 days on-site in New York City, 2 days remote.",
        "ROLE_FACTS",
    ),
    (
        "Founded in 2018, Acme Corp is revolutionizing AI-driven robotics for sustainable agriculture.",
        "COMPANY_CONTEXT",
    ),
    (
        "Our mission is to make advanced healthcare diagnostics accessible to rural communities worldwide.",
        "COMPANY_CONTEXT",
    ),
]

print("\n" + "=" * 115)
print("POSTING PIECE CLASSIFIER: LIVE SAMPLES (384-d Production vs 128-d Compressed)")
print("=" * 115)
p_header = f"{'Posting Segment Text':<54} | {'Expected':<16} | {'384-d Pred (Conf)':<20} | {'128-d Pred (Conf)':<20}"
print(p_header)
print("-" * 115)

for text, exp in posting_test_cases:
    pred_384, conf_384 = engine.classify_posting_piece(text, dim=384)
    pred_128, conf_128 = engine.classify_posting_piece(text, dim=128)
    ok_384 = "[+]" if pred_384 == exp else "[-]"
    ok_128 = "[+]" if pred_128 == exp else "[-]"
    s_384 = f"{ok_384} {pred_384} ({conf_384*100:.1f}%)"
    s_128 = f"{ok_128} {pred_128} ({conf_128*100:.1f}%)"
    print(f"{text[:53]:<54} | {exp:<16} | {s_384:<20} | {s_128:<20}")

print("=" * 115 + "\n")

