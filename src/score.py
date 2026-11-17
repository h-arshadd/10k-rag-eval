"""
STEP 2 - score saved answers against your gold answers. No LLM is used anywhere; every check is a rule you can read.

    python src/score.py runs/baseline
    python src/score.py runs/baseline runs/new_chunks    # side by side + which questions flipped

Writes runs/<name>/report.csv (one row per question) and summary.json.

What is measured
  RETRIEVAL (did the right pages come back?)   - uses the chunks saved by run_rag.py
    hit@k      : at least one gold page appears in the top-k chunks
    recall@k   : share of the gold pages that appear in the top-k (matters for questions needing 2+ pages)
    MRR        : 1 / rank of the first chunk from a gold page (0 if none in the top 10)
    facts_in_context@5 : every key number of the gold answer appears somewhere in the top-5 chunks
  GENERATION (is the answer right?)
    accuracy   : every key number in the gold answer appears in the model's answer
    unanswerable : the model must say it doesn't know (phrases listed in ABSTAIN)
    cited_file / cited_page : the answer mentions a gold file / gold page number
  Questions whose gold answer has no numbers, or more than MAX_FACTS numbers (long free text), are marked "manual": read them yourself.
"""
import re, sys, json, csv
from pathlib import Path

KS = [1, 3, 5, 10]
MAX_FACTS = 8   # gold answers with more numbers than this are long free text -> manual review, not auto-scored
# phrases that count as "I don't know". Add more here if your model words refusals differently.
ABSTAIN = ["i dont know", "i don't know", "i do not know", "does not contain", "doesn't contain",
           "not in the context", "not provided in the context", "not mentioned in the context",
           "cannot find", "can't find"]


def norm(s):
    s = s.lower().replace("\u2019", "'").replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)   # 281,724 -> 281724
    s = re.sub(r"(\d)\.00\b", r"\1", s)    # 95.00   -> 95
    s = re.sub(r"\s+%", "%", s)            # 44.8 %  -> 44.8%
    return s


def key_facts(gold):
    """numbers from the gold answer that must show up: decimals, percents, and 3+ digit numbers (not years)."""
    g = norm(re.sub(r"\((page|printed|pdf)[^)]*\)", "", gold, flags=re.I))
    facts = []
    for t in re.findall(r"\d+(?:\.\d+)?%?", g):
        core = t.rstrip("%")
        if "." in t or "%" in t or (len(core) >= 3 and not re.fullmatch(r"(19|20)\d\d", core)):
            facts.append(t)
    return sorted(set(facts))


def has_fact(text_norm, fact):
    return re.search(r"(?<![\d.])" + re.escape(fact) + r"(?!\d)", text_norm) is not None


def score_one(q, rec):
    kind = "unanswerable" if q["unanswerable"] else ("multi_source" if q["multi_source"] else "single")
    row = {"id": q["id"], "company": q["company"], "fiscal_year": q["fiscal_year"], "kind": kind,
           "status": "", "failure_reason": "", "question": q["question"], "gold_answer": q["gold_answer"],
           "model_answer": "", **{f"hit@{k}": None for k in KS}, **{f"recall@{k}": None for k in KS},
           "mrr": None, "facts_in_context@5": None, "cited_file": None, "cited_page": None}
    if rec is None or rec.get("error") or rec.get("answer") is None:
        row["status"] = "error" if rec else "missing"
        row["failure_reason"] = (rec or {}).get("error", "not run")
        return row
    ans = rec["answer"]
    row["model_answer"] = ans.replace("\n", " ")
    ans_n = norm(ans)
    abstained = any(p in ans_n for p in ABSTAIN)

    if q["unanswerable"]:
        row["status"] = "pass" if abstained else "fail"
        if not abstained:
            row["failure_reason"] = "answered instead of saying it doesn't know"
        return row

    # --- retrieval ---
    ev = {(e["file"], e["page"]) for e in q["evidence"]}
    got = [(c["file"], c["page"]) for c in rec.get("retrieved", [])]
    if ev and got:
        for k in KS:
            top = set(got[:k])
            row[f"hit@{k}"] = bool(top & ev)
            row[f"recall@{k}"] = len(top & ev) / len(ev)
        first = next((i for i, g in enumerate(got, 1) if g in ev), None)
        row["mrr"] = 1 / first if first else 0.0
        row["cited_file"] = any(f in ans for f, _ in ev)
        row["cited_page"] = any(re.search(rf"(?:page|p\.?)\s*{p}\b", ans, re.I) for _, p in ev)

    # --- generation ---
    facts = key_facts(q["gold_answer"])
    if not facts or len(facts) > MAX_FACTS:
        row["status"] = "manual"
        return row
    if rec.get("retrieved"):
        top5 = norm(" ".join(c["text"] for c in rec["retrieved"][:5]))
        row["facts_in_context@5"] = all(has_fact(top5, f) for f in facts)
    ok = all(has_fact(ans_n, f) for f in facts)
    row["status"] = "pass" if ok else "fail"
    if not ok:
        if not rec.get("retrieved"):
            row["failure_reason"] = "generation: wrong/missing numbers (no retrieved chunks were saved to diagnose)"
        elif row["hit@5"] is False:
            row["failure_reason"] = "retrieval: no gold page in top 5"
        elif row["facts_in_context@5"] is False:
            row["failure_reason"] = "context: right page retrieved but the numbers are not all in the top-5 chunks"
        elif abstained:
            row["failure_reason"] = "generation: said it doesn't know although the numbers were in the context"
        else:
            row["failure_reason"] = "generation: numbers were in the context but missing/wrong in the answer"
    return row


def mean(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    return round(sum(v) / len(v), 3) if v else None


def summarize(rows):
    answerable = [r for r in rows if r["kind"] != "unanswerable"]
    scored = [r for r in answerable if r["status"] in ("pass", "fail")]
    unans = [r for r in rows if r["kind"] == "unanswerable" and r["status"] in ("pass", "fail")]
    s = {"questions": len(rows), "scored_answerable": len(scored), "unanswerable": len(unans),
         "manual_review": sum(r["status"] == "manual" for r in rows),
         "errors": sum(r["status"] in ("error", "missing") for r in rows),
         "accuracy": mean([{**r, "x": r["status"] == "pass"} for r in scored], "x"),
         "unanswerable_correctly_refused": mean([{**r, "x": r["status"] == "pass"} for r in unans], "x")}
    for k in KS:
        s[f"hit@{k}"] = mean(answerable, f"hit@{k}")
    for k in KS:
        s[f"recall@{k}"] = mean(answerable, f"recall@{k}")
    s["mrr"] = mean(answerable, "mrr")
    s["facts_in_context@5"] = mean(answerable, "facts_in_context@5")
    s["cited_file"] = mean(answerable, "cited_file")
    s["cited_page"] = mean(answerable, "cited_page")
    for group, vals in (("company", ("MSFT", "NKE")), ("kind", ("single", "multi_source"))):
        for v in vals:
            sub = [r for r in scored if r[group] == v]
            s[f"accuracy[{v}]"] = mean([{**r, "x": r["status"] == "pass"} for r in sub], "x")
    reasons = {}
    for r in rows:
        if r["failure_reason"]:
            reasons[r["failure_reason"]] = reasons.get(r["failure_reason"], 0) + 1
    s["failure_reasons"] = reasons
    return s


def load_run(run_dir, questions):
    answers = {a["id"]: a for a in json.load(open(Path(run_dir) / "answers.json", encoding="utf-8"))}
    rows = [score_one(q, answers.get(q["id"])) for q in questions]
    summary = summarize(rows)
    with open(Path(run_dir) / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(Path(run_dir) / "summary.json", "w"), indent=2)
    return rows, summary


def pct(x):
    return "  -  " if x is None else f"{x * 100:5.1f}%"


if __name__ == "__main__":
    runs = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not runs:
        sys.exit(__doc__)
    questions = json.load(open("eval/questions.json", encoding="utf-8"))
    results = {r: load_run(r, questions) for r in runs}

    metrics = ["accuracy", "accuracy[single]", "accuracy[multi_source]", "accuracy[MSFT]", "accuracy[NKE]",
               "unanswerable_correctly_refused", "hit@1", "hit@3", "hit@5", "hit@10", "recall@5", "recall@10",
               "mrr", "facts_in_context@5", "cited_file", "cited_page"]
    print(f"{'metric':34}" + "".join(f"{Path(r).name[:16]:>18}" for r in runs))
    for m in metrics:
        print(f"{m:34}" + "".join(f"{pct(results[r][1][m]):>18}" for r in runs))
    for r in runs:
        s = results[r][1]
        print(f"\n{r}: {s['scored_answerable']} scored answerable, {s['unanswerable']} unanswerable, "
              f"{s['manual_review']} need manual review, {s['errors']} errors")
        for reason, n in sorted(s["failure_reasons"].items(), key=lambda x: -x[1]):
            print(f"   {n:3d} x {reason}")

    if len(runs) == 1:
        rows = results[runs[0]][0]
        print("\nFAILED:")
        for r in rows:
            if r["status"] == "fail":
                print(f"  {r['id']}: {r['failure_reason']}\n     Q: {r['question'][:100]}\n     model: {r['model_answer'][:160]}")
        print("\nREAD THESE YOURSELF (free-text gold answers, not auto-scored):")
        for r in rows:
            if r["status"] == "manual":
                print(f"  {r['id']}: {r['question'][:90]}\n     model: {r['model_answer'][:200]}")
    elif len(runs) == 2:
        a, b = (dict((r["id"], r["status"]) for r in results[x][0]) for x in runs)
        print(f"\nCHANGED between {runs[0]} and {runs[1]}:")
        for i in a:
            if a[i] != b[i] and {a[i], b[i]} <= {"pass", "fail"}:
                print(f"  {i}: {a[i]} -> {b[i]}")
