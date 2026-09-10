from __future__ import annotations

from typing import Iterable


def passage_context(language: str, passages: Iterable[str]) -> str:
    return "\n\n".join(
        "To help with the translation, here is a passage retrieved from a "
        f"{language}-English grammar book:\n{passage}"
        for passage in passages
    )


def long_context(language: str, book_text: str) -> str:
    return (
        "To help with the translation, here is the full text of a "
        f"{language}-English grammar book:\n"
        "—\n"
        f"{book_text}\n"
        f"This is the end of the {language}-English grammar book.\n"
        "—"
    )


def appendix_c_prompt(language: str, location: str, source: str, grammar_context: str, refusal: bool = False) -> str:
    instruction = "Now write the translation."
    if refusal:
        instruction += (
            " If you are not sure what the translation should be, then give your best guess. "
            f"Do not say that you do not speak {language}. If your translation is wrong, "
            "that is fine, but provide a translation."
        )
    return (
        f"{language} is a language spoken in {location}. Translate the following sentence "
        f"from {language} to English: {source}\n\n"
        f"{grammar_context}\n\n"
        f"{instruction}\n"
        f"{language}: {source}\n"
        "English translation:"
    )


REFUSAL_PATTERNS = (
    "i cannot translate",
    "i can't translate",
    "unable to translate",
    "do not speak",
    "don't speak",
    "not enough information",
    "cannot provide a translation",
)


def is_refusal(text: str) -> bool:
    folded = text.casefold()
    return not text.strip() or any(pattern in folded for pattern in REFUSAL_PATTERNS)


def extract_translation(text: str) -> str:
    value = text.strip()
    if value.casefold().startswith("english translation:"):
        value = value.split(":", 1)[1].strip()
    return value

