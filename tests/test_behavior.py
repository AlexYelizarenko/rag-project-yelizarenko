import os

import pytest
from dotenv import load_dotenv
from groq import APIError

from rag.evaluate import FOLD
from retrieval.index import INDEX_DIR, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")
pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY"),
    pytest.mark.skipif(not INDEX_DIR.exists(), reason="Chroma index not built"),
]


@pytest.fixture(scope="module")
def rag():
    from rag.ask import RAG
    return RAG()


def ask(rag, query):
    try:
        return rag.answer(query)
    except APIError as error:  # rate limit or outage: skip instead of failing the suite
        pytest.skip(f"Groq unavailable: {error.message}")


def cited(result):
    return {c["chunk_id"] for c in result["citations"]}


def text(result):
    return result["answer"].translate(FOLD).lower()


# Minimum Functionality: simple facts that are in the corpus, and a plain refusal.

def test_mft_definition_of_deployer(rag):
    result = ask(rag, "What is a deployer under the AI Act?")
    assert result["status"] == "ok"
    assert "article_3_def_4" in cited(result)
    assert "authority" in text(result) and "non professional" in text(result)


def test_mft_ai_literacy_obligation(rag):
    result = ask(rag, "Do companies have to train their staff who work with AI?")
    assert result["status"] == "ok"
    assert "article_4" in cited(result)
    assert "ai literacy" in text(result)


def test_mft_refuses_out_of_scope_question(rag):
    result = ask(rag, "What is a good recipe for a chocolate cake?")
    assert result["status"] == "insufficient_context"
    assert result["citations"] == []


# Invariance: surface perturbations (case, typos, filler) must not change what is cited.

@pytest.mark.parametrize("variant", [
    "WHAT IS A DEPLOYER UNDER THE AI ACT?",
    "what is a deployr under teh AI act",
    "Hi! Quick question please: what is a deployer under the AI Act? Thanks a lot.",
])
def test_inv_deployer_question_survives_surface_changes(rag, variant):
    result = ask(rag, variant)
    assert result["status"] == "ok"
    assert "article_3_def_4" in cited(result)


# Directional Expectation: a targeted change in the question must move the answer in a known direction.

def test_dir_certificate_validity_follows_the_annex(rag):
    annex_iii = ask(rag, "How long is a notified body certificate valid for an AI system covered by Annex III?")
    annex_i = ask(rag, "How long is a notified body certificate valid for an AI system covered by Annex I?")
    assert annex_iii["status"] == annex_i["status"] == "ok"
    assert "article_44" in cited(annex_iii) and "article_44" in cited(annex_i)
    assert "four years" in text(annex_iii) or "4 years" in text(annex_iii)
    assert "five years" in text(annex_i) or "5 years" in text(annex_i)


def test_dir_sme_fine_switches_to_the_lower_amount(rag):
    base = ask(rag, "What is the maximum fine for a company that breaks the obligations of deployers?")
    sme = ask(rag, "What is the maximum fine for a small start-up (an SME) that breaks the obligations of deployers?")
    assert base["status"] == sme["status"] == "ok"
    assert "15 000 000" in text(base) or "15 million" in text(base)
    assert "higher" in text(base)
    assert "lower" in text(sme)
