# 10-K RAG + eval

A small RAG over Microsoft and Nike 10-K filings, plus a rule-based harness to measure it.
(Temporary notes: where things stand so far.)

## The RAG

PDF -> PyMuPDF pages -> 1000-char chunks (150 overlap) -> `BAAI/bge-small-en-v1.5` embeddings -> Chroma -> top-5 chunks -> Groq `openai/gpt-oss-120b`.
The prompt says: answer only from the context, cite filename + page, otherwise reply `I dont know.`

**v2 change:** `src/retrieve.py` reads the company and fiscal year(s) from the question and searches only those filings
(company missing -> both companies for that year; year missing -> all years for that company; neither -> search everything).

## Layout

```
data/raw/MSFT/MSFT_FY20xx_10K.pdf     (not in git)
data/raw/NKE/NKE_FY20xx_10K.pdf       (not in git)
eval/questions.json                   52 questions with gold answers + gold pages
src/ingest.py                         load PDFs, chunk
src/embed.py                          build the Chroma index
src/retrieve.py                       filing-aware search (v2)
src/chain.py                          retrieve -> prompt -> LLM
src/run_rag.py                        run every question, save answers + top-10 chunks
src/score.py                          score saved answers (no LLM judge)
runs/<name>/                          answers.json, report.csv, summary.json per run
```

## Run it (from the repo root)

```
pip install -r requirements.txt       # plus a .env with GROQ_API_KEY=...
python src/embed.py                   # once, or after changing PDFs / chunking / embedding model
python src/run_rag.py --name v2-filter
python src/score.py runs/v2-filter
python src/score.py runs/baseline runs/v2-filter     # side by side + what flipped
```

Re-scoring saved answers is instant and does not call the model, so changing `questions.json` or `score.py` never needs a new run.

## questions.json

Each entry: `id, company, fiscal_year, question, gold_answer, file, page, evidence[{file,page}], unanswerable, multi_source, required`.
- `evidence`: every gold page (used for recall@k).
- `required` (optional): the facts a correct answer must contain, i.e. only what the question asks for.
  If absent, the numbers in `gold_answer` are used.
- 7 questions are unanswerable on purpose: the right reply is "I dont know".

## How scoring works

Retrieval (from the saved top-10 chunks): hit@k, recall@k, MRR, and facts_in_context@5 (all required facts appear in the top-5 chunks).
Generation: accuracy (every required fact appears in the answer), unanswerable refused correctly, cited_file / cited_page.
Each failure is labeled **retrieval** (no gold page in top 5), **context** (right page, facts not all in the chunks) or **generation** (facts were in context, answer missed them).
Free-text gold answers with no checkable facts are listed for manual review.

## Results so far

Scored with the original auto-extracted numbers; scoring has since changed (the `required` lists), so re-score before comparing.

| | baseline | v2-filter |
|---|---|---|
| accuracy | 57.5% | 57.5% |
| hit@5 | 50.0% | 64.3% |
| hit@10 | 59.5% | 69.0% |
| MRR | 38.2% | 50.4% |
| facts_in_context@5 | 65.0% | 80.0% |
| cited_file | 78.6% | 97.6% |
| unanswerable refused | 100% | 100% |
| failures: retrieval / generation | 12 / 4 | 6 / 9 |

- Baseline's main problem was wrong-year retrieval: 11 near-identical filings in one index and no company/year in the chunk text.
- The filter fixed much of that. Accuracy stayed flat because failures moved from retrieval to generation, and some gold answers asked for more than the question did.
- Still failing: the Nike "how many employees" questions answer `I dont know` even inside the correct filing.
  In an offline test, keyword search (BM25) inside the filtered filing got hit@5 71.4% / hit@10 85.7% vs the dense 64.3% / 69.0%, but also did not rank those employee pages well.

## Known limits

- 52 questions is small: a couple of flipped answers moves accuracy ~5 points. Judge changes mostly by the retrieval metrics.
- Number matching is strict: `$281.7 billion` will not match a required `281,724`.
- A rule can confirm the asked-for facts are present but not that extra claims are right; spot-check passes.
- The v2 run also set `temperature=0`, so it differs from baseline in two ways.

## Next

1. Re-score baseline and v2-filter with the updated `score.py` / `questions.json`.
2. Hybrid retrieval (dense + BM25, reciprocal rank fusion).
3. Look at the employee-count pages and the split table chunks (MSFT FY2021 segments, MSFT FY2025 cash flow).