"""A small bigram generator adapted from the Week 2 word sampling practical."""

from collections import Counter, defaultdict
import random
import re


class BigramModel:
    def __init__(self, corpus: list[str]):
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for sentence in corpus:
            tokens = re.findall(r"\b\w+\b", sentence.casefold())
            tokens.append("<eos>")
            for current, following in zip(tokens[:-1], tokens[1:]):
                counts[current][following] += 1
        self.counts = dict(counts)

    def generate_text(self, start_word: str, length: int) -> str:
        current = start_word.casefold().strip()
        if current not in self.counts:
            raise ValueError(f"Unknown start word: {start_word}")
        words = [current]
        for _ in range(length - 1):
            following = self.counts.get(current)
            if not following:
                break
            current = random.choices(
                list(following), weights=list(following.values()), k=1
            )[0]
            if current == "<eos>":
                break
            words.append(current)
        return " ".join(words)
