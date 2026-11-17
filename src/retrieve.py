"""
Retrieval with a filing filter (the v2 change).

Why: your index holds 11+ near-identical 10-Ks, and the chunk text doesn't say which company/year it is from,
so "fiscal 2024 employees" pulls pages from the wrong year. This reads the company and fiscal year(s) from the
question and searches only those filings. If the question names neither, it searches everything (same as before).
"""
import re
from embed import get_vector_store

vector_store = get_vector_store()

FIRST_YEAR, LAST_YEAR = 2021, 2026   # fiscal years you have PDFs for
COMPANIES = {"MSFT": r"\b(microsoft|msft)\b", "NKE": r"\b(nike|nke)\b"}


def detect_filings(query):
    """-> list of filenames to search, e.g. ['NKE_FY2024_10K.pdf'], or None if the question gives neither company nor year.
    Company unknown -> all companies for that year. Year unknown -> all years for that company."""
    companies = [c for c, pat in COMPANIES.items() if re.search(pat, query, re.I)]
    years = re.findall(r"(?:fiscal(?:\s+year)?|fy)\s*'?(20\d\d)", query, re.I)      # "fiscal 2024", "FY2024"
    if not years:
        years = re.findall(r"(?:june 30|may 31),?\s*(20\d\d)", query, re.I)           # "as of June 30, 2025"
    if not years:
        years = re.findall(r"\b(20\d\d)\b", query)                                     # any year as a last resort
    years = sorted({y for y in years if FIRST_YEAR <= int(y) <= LAST_YEAR})
    if not companies and not years:
        return None
    companies = companies or list(COMPANIES)
    years = years or [str(y) for y in range(FIRST_YEAR, LAST_YEAR + 1)]
    return [f"{c}_FY{y}_10K.pdf" for c in companies for y in years]


def search(query, k=5):
    files = detect_filings(query)
    if files:
        docs = vector_store.similarity_search(query, k=k, filter={"filename": {"$in": files}})
        if docs:
            return docs
    return vector_store.similarity_search(query, k=k)
