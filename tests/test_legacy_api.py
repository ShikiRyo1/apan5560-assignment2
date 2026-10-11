"""Exercise HTTP contracts against the real installed spaCy vectors."""

import numpy as np
import pytest
import spacy
from fastapi.testclient import TestClient

from app.embeddings import EXCLUDED_COMPONENTS, MODEL_NAME
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def reference_nlp():
    # A separate model instance is the reference, not a mocked or hardcoded vector.
    return spacy.load(MODEL_NAME, exclude=EXCLUDED_COMPONENTS)


@pytest.mark.parametrize("word", ["apple", "king", "university"])
def test_embedding_matches_spacy(client, reference_nlp, word):
    response = client.get("/embedding", params={"word": word})
    assert response.status_code == 200
    data = response.json()
    assert data["normalized_word"] == word
    assert data["model"] == "en_core_web_lg"
    assert data["model_version"] == "3.8.0"
    assert data["dimensions"] == len(data["vector"]) == 300
    assert np.isfinite(data["vector"]).all()
    expected = reference_nlp(word)[0].vector
    np.testing.assert_allclose(data["vector"], expected, rtol=0, atol=0)


def test_normalization_and_model_reuse(client):
    initial_model = client.app.state.embeddings.nlp
    original = client.get("/embedding", params={"word": "apple"}).json()
    normalized = client.get("/embedding", params={"word": "  APPLE  "}).json()
    assert normalized["word"] == "APPLE"
    assert normalized["normalized_word"] == "apple"
    assert normalized["vector"] == original["vector"]
    assert client.app.state.embeddings.nlp is initial_model


@pytest.mark.parametrize("word", ["", "   ", "two words", "apple!", "123", "x" * 101])
def test_invalid_word_is_rejected(client, word):
    assert client.get("/embedding", params={"word": word}).status_code == 422


def test_missing_word_is_rejected(client):
    assert client.get("/embedding").status_code == 422


def test_unknown_word_does_not_return_a_fake_vector(client):
    response = client.get("/embedding", params={"word": "zzzxxyyqnotarealword"})
    assert response.status_code == 404
    assert "No pretrained vector" in response.json()["detail"]


def test_health_reports_loaded_model(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "model": "en_core_web_lg",
        "model_version": "3.8.0",
        "dimensions": 300,
    }


def test_generator_preserves_valid_bigram_transitions(client):
    response = client.post("/generate", json={"start_word": "the", "length": 5})
    assert response.status_code == 200
    words = response.json()["generated_text"].split()
    assert words[0] == "the"
    assert 1 <= len(words) <= 5
    counts = client.app.state.bigram.counts
    assert all(second in counts[first] for first, second in zip(words, words[1:]))


@pytest.mark.parametrize("length", [0, 51])
def test_generator_rejects_invalid_length(client, length):
    assert client.post("/generate", json={"start_word": "the", "length": length}).status_code == 422


def test_generator_rejects_unknown_start(client):
    assert client.post("/generate", json={"start_word": "missing", "length": 5}).status_code == 400


def test_documentation_exposes_embedding_query(client):
    assert client.get("/docs").status_code == 200
    operation = client.get("/openapi.json").json()["paths"]["/embedding"]["get"]
    assert any(p["name"] == "word" and p["required"] for p in operation["parameters"])
