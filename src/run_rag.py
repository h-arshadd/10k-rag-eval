"""
STEP 1 - run every question through your RAG and save what it did.

    python src/run_rag.py --name baseline            # all questions
    python src/run_rag.py --name baseline --limit 3  # quick smoke test

Run from the repo root (chroma_db/ and .env are found relative to it).
Saves runs/<name>/answers.json: for each question, the model's answer AND the top-10
chunks the retriever returned (file, page, chunk_id, text). Those chunks are what recall@k is computed from.
Re-running with the same --name skips questions already answered, so it resumes after an API error.
To start a run over, delete runs/<name>/answers.json.
"""
import json, time, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="label for this run, e.g. baseline or k8-chunk600")
ap.add_argument("--questions", default="eval/questions.json")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--top", type=int, default=10, help="chunks to save per question (max k for recall@k)")
ap.add_argument("--sleep", type=float, default=0.5, help="seconds between questions (API rate limits)")
args = ap.parse_args()

from chain import rag_chain  # imported after parsing so --help works without a GROQ key
from retrieve import search  # the same search the chain uses

questions = json.load(open(args.questions, encoding="utf-8"))
if args.limit:
    questions = questions[: args.limit]

out_dir = Path("runs") / args.name
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "answers.json"
done = {r["id"]: r for r in json.load(open(out_file, encoding="utf-8"))} if out_file.exists() else {}

for i, q in enumerate(questions, 1):
    if q["id"] in done and not done[q["id"]].get("error"):
        continue
    rec = {"id": q["id"], "question": q["question"]}
    try:
        # same query + same index => same ranking the chain used for its own top 5
        docs = search(q["question"], k=args.top)
        rec["retrieved"] = [
            {"rank": r, "file": d.metadata["filename"], "page": d.metadata["page"] + 1,  # PyMuPDF pages start at 0
             "chunk_id": d.metadata.get("chunk_id"), "text": d.page_content}
            for r, d in enumerate(docs, 1)
        ]
        rec["answer"] = rag_chain.invoke(q["question"]).content
    except Exception as e:  # keep going; failed ones are retried on the next run
        rec["error"] = repr(e)
    done[q["id"]] = rec
    json.dump(list(done.values()), open(out_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[{i}/{len(questions)}] {q['id']}: " + (rec.get("error") or rec["answer"][:90].replace("\n", " ")))
    time.sleep(args.sleep)

print(f"\nsaved {out_file}  ->  now run: python src/score.py runs/{args.name}")