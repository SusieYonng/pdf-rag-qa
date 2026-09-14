# PDF RAG QA

Ask questions about a folder of local PDFs. Every answer is grounded in your
documents and cites the file, page number, and exact text snippet it came from.

Built with LangChain (LCEL), Chroma, OpenAI, and Streamlit.

---

## Setup

### 1. Install

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

### 2. Configure `.env`

Copy the template and add your key:

```bash
cp .env.example .env
```

Then edit `.env` so it contains at minimum:

```
OPENAI_API_KEY=sk-...
```

Get a key from <https://platform.openai.com/api-keys>. Every other setting in
`.env.example` is optional and documented inline — models, chunk size, and paths
all have working defaults.

### 3. Add PDFs

Drop them into `./data/`. Subfolders are scanned too.

```bash
cp ~/Downloads/*.pdf data/
```

### 4. Run

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. Click **Build index** in the sidebar,
wait for embedding to finish, then ask a question.

---

## How it works

```
./data/*.pdf
  │
  ├─ PyPDFLoader ─────────────── one Document per page, with source + page metadata
  ├─ RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  ├─ OpenAIEmbeddings ────────── text-embedding-3-small
  ├─ Chroma ──────────────────── persisted to ./chroma_db
  │
  └─ LCEL retrieval chain ────── {"question"} → {"question", "context", "answer"}
```

The chain is composed from `langchain-core` runnables:

```python
RunnablePassthrough.assign(context=itemgetter("question") | retriever)
                   .assign(answer=prompt | llm | StrOutputParser())
```

Because `context` is assigned before `answer`, the retrieved `Document` objects
stay in the output alongside the generated text — that's what powers the
**Sources** panel. The same chain streams: `.stream()` emits the context once,
then answer tokens, so the UI renders text as it arrives.

**Citations.** Retrieved chunks are fed to the model as numbered passages
(`[1] report.pdf (page 4) …`), and the system prompt requires an inline `[n]`
marker for every factual claim. The numbers in the answer line up with the
entries in the Sources expander.

### Index management

The app writes a `manifest.json` next to the Chroma files recording a
fingerprint of every source PDF (name, size, mtime) plus the chunking and
embedding settings. If you add, edit, or remove a PDF, the sidebar flags the
index as stale and prompts a rebuild.

Rebuilds are wholesale rather than incremental. That costs a few more embedding
calls but guarantees a deleted or edited PDF can't leave orphaned chunks behind
that would later be cited as sources.

### Error handling

A corrupt, encrypted, or image-only PDF is reported by name in the sidebar and
skipped — one bad file never aborts the whole build. Scanned PDFs with no text
layer are called out specifically, since those need OCR before this app can use
them.

---

## Configuration reference

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | **Required.** |
| `CHAT_MODEL` | `gpt-4o-mini` | Also overridable live in the sidebar. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Changing this requires a rebuild. |
| `CHUNK_SIZE` | `1000` | Changing this requires a rebuild. |
| `CHUNK_OVERLAP` | `200` | Changing this requires a rebuild. |
| `DATA_DIR` | `./data` | Scanned recursively for `*.pdf`. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Safe to delete; just rebuild after. |
| `CHROMA_COLLECTION` | `pdf_rag` | |

`k` (passages retrieved) and temperature are sidebar controls, not env vars.

---

## Cost

Indexing is the one-off cost; queries are cheap. With
`text-embedding-3-small`, a 200-page PDF is roughly 100k tokens — a fraction of
a cent to embed. The index is reused across restarts, so you pay it once per
rebuild, not once per launch.

---

## Troubleshooting

**"OPENAI_API_KEY is not set"** — `.env` must sit next to `app.py`, and the key
needs no surrounding quotes. Restart the app after editing it.

**Answers say the context doesn't contain the answer** — raise `k` in the
sidebar so more passages are retrieved. If that doesn't help, the relevant text
may not have been extracted; check the sidebar for skipped files.

**A PDF is listed in `data/` but not in the index** — look for a red error in
the sidebar after building. Image-only scans need OCR (e.g. `ocrmypdf`) first.

**Dimension mismatch errors from Chroma** — you changed `EMBEDDING_MODEL`
against an existing index. Delete `./chroma_db` and rebuild.

---

## Dependency notes

Two things worth knowing, both verified against the pinned versions:

1. **The top-level `langchain` package is not a dependency.** In LangChain 1.x
   it became an agent framework and no longer ships `langchain.chains`, so
   `create_retrieval_chain` and `create_stuff_documents_chain` are gone. Most
   RAG tutorials you'll find online still import them and will fail on 1.x. This
   app composes the chain from `langchain-core` primitives instead, which is the
   current idiom.

2. **`langchain-community` is being sunset** and emits a `DeprecationWarning` on
   import. It is still required here because `PyPDFLoader` has no standalone
   package yet — there is no `langchain-pypdf` on PyPI as of the pin date. It
   works correctly; the warning is cosmetic. If you'd rather drop the dependency
   entirely, `load_and_split()` in `app.py` is the only place it's used and can
   be rewritten against `pypdf` directly in about ten lines.
