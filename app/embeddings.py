"""Load the spaCy model once and look up static vectors for individual words."""

import spacy

MODEL_NAME = "en_core_web_lg"
MODEL_VERSION = "3.8.0"
EXPECTED_DIMENSIONS = 300
EXCLUDED_COMPONENTS = [
    "tok2vec", "tagger", "parser", "senter", "ner", "attribute_ruler", "lemmatizer"
]


class InvalidWordError(ValueError):
    """The input is empty or does not represent one alphabetic token."""


class UnknownWordError(LookupError):
    """The model has no pretrained static vector for this word."""


class WordEmbeddingService:
    def __init__(self) -> None:
        # Tokenization and pretrained vectors are sufficient for this endpoint.
        # Excluding taggers, parsers and NER avoids unnecessary startup work.
        self.nlp = spacy.load(MODEL_NAME, exclude=EXCLUDED_COMPONENTS)
        self.dimensions = self.nlp.vocab.vectors_length
        self.model_version = self.nlp.meta["version"]
        if self.dimensions != EXPECTED_DIMENSIONS:
            raise RuntimeError(f"Expected 300-dimensional vectors, got {self.dimensions}")
        if self.model_version != MODEL_VERSION:
            raise RuntimeError(f"Expected model {MODEL_VERSION}, got {self.model_version}")

    def embed(self, word: str) -> dict:
        cleaned = word.strip()
        normalized = cleaned.casefold()
        if not normalized:
            raise InvalidWordError("Enter one non-empty word.")

        tokens = self.nlp.make_doc(normalized)
        if len(tokens) != 1 or not tokens[0].is_alpha:
            raise InvalidWordError("Enter one alphabetic word, without spaces or punctuation.")

        token = tokens[0]
        if not token.has_vector:
            raise UnknownWordError(f"No pretrained vector is available for '{normalized}'.")

        return {
            "word": cleaned,
            "normalized_word": normalized,
            "model": MODEL_NAME,
            "model_version": self.model_version,
            "dimensions": self.dimensions,
            "vector": token.vector.tolist(),
        }
