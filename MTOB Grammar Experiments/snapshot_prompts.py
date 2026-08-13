#!/usr/bin/env python3
from mtob_grammar.config import output_path
from mtob_grammar.prompts import appendix_c_prompt, long_context, passage_context


def main() -> None:
    language = "{LANGUAGE}"
    location = "{LOCATION}"
    source = "{SOURCE}"
    snapshots = {
        "ge_gs_language_to_english.txt": appendix_c_prompt(
            language,
            location,
            source,
            passage_context(language, ["{BOOK_TEXT_1}", "{BOOK_TEXT_2}"]),
        ),
        "gl_language_to_english.txt": appendix_c_prompt(
            language, location, source, long_context(language, "{GL_TEXT}")
        ),
        "refusal_retry_language_to_english.txt": appendix_c_prompt(
            language,
            location,
            source,
            passage_context(language, ["{BOOK_TEXT_1}", "{BOOK_TEXT_2}"]),
            refusal=True,
        ),
    }
    directory = output_path("manifests", "prompt_templates")
    directory.mkdir(parents=True, exist_ok=True)
    for filename, text in snapshots.items():
        (directory / filename).write_text(text + "\n", encoding="utf-8")
    print(f"[OK] Wrote {len(snapshots)} Appendix C prompt snapshots")


if __name__ == "__main__":
    main()
