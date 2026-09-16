"""
Local PDF RAG QA
================

A self-contained Streamlit application that answers questions over a folder of
local PDFs using LangChain, Chroma and OpenAI, and cites the exact page and
chunk snippet behind every answer.

Pipeline
--------
    ./data/*.pdf
        -> PyPDFLoader              (one Document per page, robust per-file)
        -> RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        -> OpenAIEmbeddings         -> Chroma (persisted on disk)
        -> LCEL retrieval chain     -> {"answer": str, "context": list[Document]}

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Iterator

import streamlit as st
from dotenv import load_dotenv
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("DATA_DIR", APP_ROOT / "data"))
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", APP_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pdf_rag")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# The manifest records what the on-disk index was built from, so the app can
# tell the user when ./data has changed and the index is stale.
MANIFEST_PATH = PERSIST_DIR / "manifest.json"

SYSTEM_PROMPT = """You are a precise research assistant answering questions about a set of PDF documents.

Rules:
- Answer using ONLY the numbered context passages below. Do not use outside knowledge.
- If the context does not contain the answer, say so plainly and state what is missing. Do not guess.
- Cite the passages you used inline with bracketed numbers, e.g. [1] or [2][4]. Every factual claim needs a citation.
- Quote short phrases verbatim when precision matters. Be concise.

Context passages:
{context}"""


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


@dataclass
class IngestReport:
    """Outcome of a corpus build, surfaced in the UI."""

    files: int = 0
    pages: int = 0
    chunks: int = 0
    failures: list[tuple[str, str]] = None  # (filename, error message)

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []


def discover_pdfs(data_dir: Path) -> list[Path]:
    """Return every PDF under `data_dir`, recursively, in stable order."""
    if not data_dir.is_dir():
        return []
    return sorted(p for p in data_dir.rglob("*.pdf") if p.is_file())


def corpus_fingerprint(paths: Iterable[Path]) -> str:
    """Hash file identity + the parameters that shape the index.

    Changing a PDF, adding one, or tuning the chunking/embedding settings all
    produce a different fingerprint, which marks the persisted index stale.
    """
    h = hashlib.sha256()
    h.update(f"{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBEDDING_MODEL}".encode())
    for path in paths:
        stat = path.stat()
        h.update(f"|{path.name}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return h.hexdigest()


def load_and_split(paths: list[Path]) -> tuple[list[Document], IngestReport]:
    """Load each PDF page-by-page and split it into overlapping chunks.

    A malformed or encrypted PDF is recorded as a failure and skipped rather
    than aborting the whole build.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split on the most semantically meaningful boundary that fits.
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    report = IngestReport()
    chunks: list[Document] = []

    for path in paths:
        try:
            pages = PyPDFLoader(str(path)).load()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            report.failures.append((path.name, str(exc)))
            continue

        # Drop pages that are entirely whitespace (scans without a text layer).
        pages = [p for p in pages if p.page_content and p.page_content.strip()]
        if not pages:
            report.failures.append(
                (path.name, "No extractable text — is this a scanned PDF needing OCR?")
            )
            continue

        for page in pages:
            page.metadata["source"] = str(path.relative_to(DATA_DIR))
            page.metadata["file_name"] = path.name

        report.files += 1
        report.pages += len(pages)
        chunks.extend(splitter.split_documents(pages))

    report.chunks = len(chunks)
    return chunks, report


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def read_manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_manifest(fingerprint: str, report: IngestReport) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "files": report.files,
                "pages": report.pages,
                "chunks": report.chunks,
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            indent=2,
        )
    )


def index_exists() -> bool:
    return bool(read_manifest().get("chunks"))


@st.cache_resource(show_spinner=False)
def open_vector_store() -> Chroma:
    """Open the persisted Chroma collection (does not build it)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIR),
    )


def release_chroma_clients() -> None:
    """Drop every cached handle on the persist directory.

    Two independent caches keep Chroma clients alive, and both must go before
    the directory can be swapped or deleted:

    * Streamlit's `@st.cache_resource` on `open_vector_store`.
    * chromadb's own `SharedSystemClient._identifier_to_system`, a
      process-global dict keyed by persist path. Constructing `Chroma()` for a
      path already in that dict reuses the existing client — including its open
      SQLite connection. If the files were replaced underneath it, queries fail
      with "no such table: tenants".
    """
    open_vector_store.clear()
    SharedSystemClient.clear_system_cache()


def build_index(paths: list[Path]) -> IngestReport:
    """Rebuild the vector store from scratch for the given PDFs.

    Rebuilding wholesale (rather than appending) keeps the index consistent
    with ./data: deleted and edited files cannot leave orphan chunks behind.

    The new index is built in a staging directory and only swapped in once it
    is complete, so a failure partway through (an API error, an interrupted
    run) leaves the previous index intact rather than destroying it.
    """
    chunks, report = load_and_split(paths)
    if not chunks:
        return report

    staging = PERSIST_DIR.with_name(PERSIST_DIR.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(staging),
        )
        # Batch the embedding calls to stay well inside OpenAI's request limits.
        for start in range(0, len(chunks), 128):
            store.add_documents(chunks[start : start + 128])
    except BaseException:
        # Leave the previous index untouched and clean up the partial build.
        release_chroma_clients()
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Swap the finished index in. Release handles first: the staging client
    # holds the files we are about to rename, and a stale client may still
    # hold the directory we are about to delete.
    release_chroma_clients()
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
    staging.rename(PERSIST_DIR)

    write_manifest(corpus_fingerprint(paths), report)
    return report


# --------------------------------------------------------------------------- #
# Retrieval chain
# --------------------------------------------------------------------------- #


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks as numbered passages the model can cite."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(
            f"[{i}] {doc.metadata.get('file_name', 'unknown')} "
            f"(page {display_page(doc)})\n{doc.page_content}"
        )
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


def display_page(doc: Document) -> str:
    """Human-facing page number: pypdf's `page` is 0-indexed."""
    label = doc.metadata.get("page_label")
    if label:
        return str(label)
    page = doc.metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "?"


@st.cache_resource(show_spinner=False)
def get_llm(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def build_rag_chain(k: int, model: str, temperature: float) -> Runnable:
    """Compose the RAG chain.

    Invoking it with {"question": ...} yields {"question", "context", "answer"},
    where `context` holds the Documents that produced the answer, so the UI can
    show citations. The chain also streams: `.stream()` emits {"context": ...}
    once, then {"answer": <token>} chunks.
    """
    retriever = open_vector_store().as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_context(x["context"]))
        | prompt
        | get_llm(model, temperature)
        | StrOutputParser()
    )

    return RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
    ).assign(answer=answer_chain)


def stream_answer(chain: Runnable, question: str) -> Iterator[str]:
    """Yield answer tokens, stashing retrieved documents in session state."""
    st.session_state.last_context = []
    for part in chain.stream({"question": question}):
        if "context" in part:
            st.session_state.last_context = part["context"]
        if "answer" in part:
            yield part["answer"]


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def render_sources(docs: list[Document]) -> None:
    if not docs:
        return
    with st.expander(f"Sources ({len(docs)} passages)", expanded=False):
        for i, doc in enumerate(docs, start=1):
            st.markdown(
                f"**[{i}] {doc.metadata.get('file_name', 'unknown')}** "
                f"— page {display_page(doc)}"
            )
            snippet = doc.page_content.strip()
            if len(snippet) > 700:
                snippet = snippet[:700].rstrip() + " …"
            st.caption(snippet)
            if i < len(docs):
                st.divider()


def render_sidebar() -> tuple[int, str, float]:
    with st.sidebar:
        st.header("Corpus")

        pdfs = discover_pdfs(DATA_DIR)
        manifest = read_manifest()

        if not pdfs:
            st.warning(f"No PDFs found in `{DATA_DIR.name}/`. Add some and refresh.")
        else:
            st.caption(f"{len(pdfs)} PDF(s) in `{DATA_DIR.name}/`")
            with st.expander("Files", expanded=False):
                for p in pdfs:
                    st.write(f"• {p.name}")

        if manifest.get("chunks"):
            st.success(
                f"Index: {manifest['chunks']} chunks from "
                f"{manifest['files']} file(s), {manifest['pages']} page(s)"
            )
            if pdfs and manifest.get("fingerprint") != corpus_fingerprint(pdfs):
                st.warning("`data/` has changed since the last build — rebuild to sync.")
        else:
            st.info("No index yet — build one to start asking questions.")

        if st.button(
            "Rebuild index" if manifest.get("chunks") else "Build index",
            type="primary",
            disabled=not pdfs,
            use_container_width=True,
        ):
            report = None
            try:
                with st.spinner("Loading, splitting and embedding PDFs…"):
                    report = build_index(pdfs)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                st.error(f"Build failed: {exc}")
                st.caption("The previous index was left untouched.")

            if report is not None:
                for name, err in report.failures:
                    st.error(f"{name}: {err}")
                if report.chunks:
                    st.success(f"Indexed {report.chunks} chunks.")
                    st.session_state.messages = []
                    st.rerun()
                elif not report.failures:
                    st.error("Nothing to index — no extractable text found.")

        st.header("Retrieval")
        k = st.slider("Passages retrieved (k)", 1, 12, 4)

        st.header("Model")
        model = st.text_input("Chat model", value=CHAT_MODEL)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

        st.divider()
        st.caption(
            f"Embeddings: `{EMBEDDING_MODEL}` · "
            f"chunk {CHUNK_SIZE} / overlap {CHUNK_OVERLAP}"
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return k, model, temperature


def main() -> None:
    st.set_page_config(page_title="PDF RAG QA", page_icon="📄", layout="wide")
    st.title("📄 PDF RAG QA")
    st.caption("Ask questions about your local PDFs — every answer cites its sources.")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_context", [])

    k, model, temperature = render_sidebar()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("`OPENAI_API_KEY` is not set. Add it to a `.env` file and restart.")
        st.stop()

    if not index_exists():
        st.info(
            f"Drop PDFs into `{DATA_DIR.name}/`, then click **Build index** "
            "in the sidebar."
        )
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chain = build_rag_chain(k, model, temperature)
        try:
            answer = st.write_stream(stream_answer(chain, question))
        except Exception as exc:  # noqa: BLE001 - network/API errors reach the user
            st.error(f"Query failed: {exc}")
            st.session_state.messages.pop()
            return

        sources = st.session_state.last_context
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
