from __future__ import annotations

import re
from pathlib import Path

import fitz

from create_grammar_context_materials import render_markdown_to_jpg, render_pdf_page, write_both


LANGUAGE = "Tsez"
PDF_PATH_OR_FOLDER = Path("./Tsez Grammar PDF")
OUTPUT_DIR = Path(".")
MATERIALS_ROOT = OUTPUT_DIR / "Tsez Grammar Materials"

KEYWORD_WEIGHTS = [
    (5, re.compile(r"\b(table|paradigm|case|absolutive|ergative|genitive|lative|local cases)\b", re.I)),
    (5, re.compile(r"\b(gloss|translation|\(\d+\)|example|ABS|ERG|LAT|GEN)\b")),
    (4, re.compile(r"\b(pronoun|agreement|gender|number|tense|aspect|mood|evidentiality)\b", re.I)),
    (4, re.compile(r"\b(negation|negative|question|interrogative|imperative|optative)\b", re.I)),
    (3, re.compile(r"\b(word order|head-final|clause|relative|complement|adverbial|converb)\b", re.I)),
    (3, re.compile(r"\b(postposition|particle|focus|suffix|prefix|morpheme|causative|potential)\b", re.I)),
    (2, re.compile(r"\b(abbreviations|Roman numerals|genders)\b", re.I)),
    (-4, re.compile(r"\b(references|bibliography)\b", re.I)),
]


SELECTED_PAGES = [8, 9, 10, 12, 15, 16, 20, 21, 25]


def find_pdfs(path_or_folder: Path) -> list[Path]:
    if path_or_folder.is_file() and path_or_folder.suffix.lower() == ".pdf":
        return [path_or_folder]
    return sorted(path_or_folder.glob("*.pdf"))


def page_texts(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    return [page.get_text("text").strip() for page in doc]


def score_page(text: str) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    for weight, pattern in KEYWORD_WEIGHTS:
        if pattern.search(text):
            score += weight
            label = pattern.pattern.replace("\\b", "").replace("(", "").replace(")", "")
            reasons.append(f"{weight:+d} {label[:48]}")
    return score, reasons


def first_lines(text: str, n: int = 9) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines[:n])


def selected_reason(page_number: int) -> str:
    reasons = {
        8: "gender system; number and case overview",
        9: "nonlocal cases and local case table start",
        10: "local case paradigms and spatial nouns",
        12: "pronouns; adjective/linker material",
        13: "demonstrative and interrogative pronouns",
        14: "pronoun/demonstrative tables; verb section start",
        15: "verb morphological groups; TAM/evidentiality",
        16: "imperative, negation, non-finite forms",
        17: "converbs, infinitive, verbal noun, potential/causative",
        20: "postpositions; head-final word order",
        21: "ergative alignment and gender agreement",
        23: "ditransitives and causatives",
        25: "questions",
        27: "relative clauses",
        29: "adverbial clauses with converbs",
        31: "sentential and constituent negation",
        39: "glossing abbreviation list start",
        40: "glossing abbreviation list end",
    }
    return reasons.get(page_number, "translation-relevant grammar")


def usefulness(page_number: int) -> str:
    return "high" if page_number in {8, 9, 10, 15, 16, 20, 21, 25, 27, 29, 31, 39, 40} else "medium"


def abbrev_table() -> str:
    rows = [
        ("ABS", "Absolutive", "intransitive subject / transitive object role cue", "PDF p. 39"),
        ("ERG", "Ergative", "transitive subject/agent case cue", "PDF p. 39"),
        ("GEN1 / GEN2", "Genitive forms", "possessor/linking relations", "PDF p. 39"),
        ("LAT", "Lative", "to/toward; also experiencer with affective verbs", "PDF p. 39"),
        ("ESS / ABL / VERS", "Essive/Ablative/Versative", "at/from/towards local case dimensions", "PDF p. 39-40"),
        ("IN / CONT / SUPER / SUB / AD / APUD / POSS", "localization series", "in/among/on/under/at/near/on vertical", "PDF p. 39-40"),
        ("PSTWIT / PSTUNW", "past witnessed/unwitnessed", "evidential distinction in past tense", "PDF p. 40"),
        ("PRS", "present", "present tense", "PDF p. 40"),
        ("PTCP / PSTPTCP / PRSPTCP", "participle labels", "relative clause and modifier forms", "PDF p. 39-40"),
        ("PFVCVB / IPFVCVB / ANTCVB / CAUSCVB / LOCCVB", "converb labels", "adverbial clause/chaining cues", "PDF p. 39-40"),
        ("NEG", "negation", "negative form/particle cue", "PDF p. 40"),
        ("Q", "interrogative", "question suffix/function cue", "PDF p. 40"),
        ("POT", "potential", "ability/potential construction", "PDF p. 40"),
        ("PROH", "prohibitive", "negative command", "PDF p. 40"),
        ("QUOT", "quotative", "reported speech/complement cue", "PDF p. 40"),
        ("I-IV / nI", "gender markers", "noun class agreement; nI means non-I", "PDF p. 40"),
    ]
    out = ["| Form | Function | Translation cue | Source page |", "|---|---|---|---|"]
    out.extend(f"| {a} | {b} | {c} | [{d}] |" for a, b, c, d in rows)
    return "\n".join(out)


def summary_content(pdf_name: str) -> str:
    return f"""
# Tsez Grammar Summary for Translation

## Language and source PDF
| Item | Value |
|---|---|
| Language | Tsez |
| Source PDF | {pdf_name} |
| Source limitation | Embedded PDF text was available. The PDF is a 40-page grammar sketch/excerpt. No outside grammar facts were added. |
| Extraction notes | Several phonetic symbols and affricate/pharyngealization symbols occur. Forms are preserved as extracted; selected page JPGs should be checked for exact typography. |

## TRANSLATION_CUE overview
| Label | Most useful cue | Source page |
|---|---|---|
| WORD_ORDER | Tsez is consistently head-final; dependent clauses precede the independent clause, but predicate position is often clause-medial. | [PDF p. 20] |
| CASE | Tsez is morphologically ergative; intransitive subject and transitive object are absolutive; transitive agent is ergative. | [PDF p. 21-23] |
| AGREEMENT | Predicate and some modifiers/adverbs agree with the absolutive noun phrase in gender/number when agreement is visible. | [PDF p. 21] |
| QUESTION | Yes-no questions use interrogative suffix -(j)a: on the focused constituent. | [PDF p. 25] |
| NEGATION | Sentential negation uses negative verb forms; constituent negation uses particle a:nu after the negated constituent. | [PDF p. 31] |
| SUBORDINATION | Relative clauses use participles; adverbial clauses commonly use converbs. | [PDF p. 27], [PDF p. 29] |

## Typological overview
| Fact | Translation cue | Source page |
|---|---|---|
| Tsez belongs to the Tsezic group within the Nakh-Daghestanian language family. | Background only; not a translation cue. | [PDF p. 1] |
| The language is officially unwritten, though adaptations of Avar Cyrillic are used. | Orthographic forms may vary; this PDF uses a linguistic transcription. | [PDF p. 2], [PDF p. 4-5] |
| Roman numerals I-IV are used for genders; nI means non-I. | Use gender markers as agreement/class cues. | [PDF p. 40] |

## WORD_ORDER
| Fact | Translation cue | Source page |
|---|---|---|
| Tsez is consistently head-final. | Expect postpositions, prenominal relative clauses, adjectives, genitives, numerals, and dependent-before-main clause order. | [PDF p. 20] |
| The predicate position is often clause-medial. | Do not force every clause into final-verb order in translation analysis. | [PDF p. 20] |
| Noun phrase modifiers follow an ordered template: relative clause, possessive pronouns, restrictive adjective, demonstrative, numeral. | Prenominal material can correspond to English relative/possessive/adjectival modifiers. | [PDF p. 19-20] |
| Adpositional phrases are always head-final. | Postpositions translate as English prepositions such as 'before', 'after', 'near', 'because of'. | [PDF p. 20] |

## NOUNS_AND_CASE
| Form/category | Function | Translation cue | Source page |
|---|---|---|---|
| Number | singular/plural distinction | plural can be suffixal; check noun stem alternations. | [PDF p. 8] |
| Gender I-IV | noun class/gender | agreement prefixes on some adjectives, adverbs, verbs, postpositions, and one particle. | [PDF p. 7-8] |
| ABS -Ø | absolutive | intransitive subject and transitive object role cue. | [PDF p. 9], [PDF p. 21-23] |
| ERG -a: | ergative | transitive agent/subject role cue. | [PDF p. 9], [PDF p. 21-23] |
| GEN1 -s / GEN2 -zo | genitives | possessor/modifier; two forms relate to limited case concord. | [PDF p. 9] |
| DAT -z / LAT -r | dative/lative-type roles | goal and experiencer-like roles; lative experiencer in affective clauses. | [PDF p. 9], [PDF p. 24] |
| IN, CONT, SUPER, SUB, AD, APUD, POSS localizations | local case series | in/among/on/under/at/near/on-vertical localization cues. | [PDF p. 9-10] |
| ESS, LAT, ABL, VERS dimensions | local case dimensions | at/to/from/towards dimensions. | [PDF p. 9-10] |
| Distal local cases | behind/distal local meanings | distal forms add -a:z material in table. | [PDF p. 10] |
| Spatial nouns | intrinsic localization | directional suffixes attach directly, e.g. idu-r '(to) home'. | [PDF p. 10] |

## PRONOUNS
| Area | Function | Translation cue | Source page |
|---|---|---|
| Personal pronouns | first and second person only as personal pronouns | third person is expressed by demonstratives že 'he, she, it', žedi 'they'. | [PDF p. 12] |
| 1SG/2SG pronouns | unusual single form used across several case-like contexts | avoid over-segmenting; check page text/table for exact form. | [PDF p. 12] |
| 1PL/2PL pronouns | regular absolutive/ergative distinction and gender distinction in oblique stems | pronoun form may encode case/gender class. | [PDF p. 12] |
| Interrogative pronouns | no humanness distinction in absolutive; irregular ergative forms noted | wh-forms may move or remain in situ depending on focus/exhaustivity. | [PDF p. 13], [PDF p. 25-26] |

## Demonstratives, definiteness, determiners
| Form/fact | Function | Translation cue | Source page |
|---|---|---|
| Demonstratives serve as third-person pronouns. | translate demonstratives as he/she/it/they where context requires. | [PDF p. 12-13] |
| Demonstrative pronouns distinguish gender I vs II-IV in oblique forms; proximal demonstratives also in absolutive singular. | demonstrative form may signal noun class. | [PDF p. 13] |
| Attributive demonstratives have simplified absolutive vs oblique opposition. | demonstrative case marking is less rich than noun case marking. | [PDF p. 7], [PDF p. 13] |

## VERB morphology
| Area | Function | Translation cue | Source page |
|---|---|---|
| Verb classes | four morphological groups by stem-final segment. | suffix shape/stem alternation depends on verb group. | [PDF p. 15] |
| Tense-mood-aspect-evidentiality | verbs obligatorily mark these categories except in imperatives and some non-finite forms. | TAM/evidentiality is central for English tense/aspect rendering. | [PDF p. 6], [PDF p. 15] |
| Past witnessed vs past unwitnessed | evidential past distinction. | choose English past; record evidential nuance if relevant. | [PDF p. 15], [PDF p. 40] |
| Present / future / progressive | present, future indefinite/definite, progressive forms discussed. | translate tense/aspect according to suffix/context. | [PDF p. 15] |
| Imperative | second-person imperative has zero suffix for intransitives and derived transitives; -o for simple transitives. | command form. | [PDF p. 16] |
| Negation suffix -č'V | basic verbal negation with several irregularities. | negative verb forms may not be transparent. | [PDF p. 16] |
| Non-finite forms | participles, converbs, infinitive, verbal noun/masdar. | supports relative, adverbial, complement, and purpose translations. | [PDF p. 16-17], [PDF p. 30] |
| Causative -r- | derives transitive verbs from intransitive/affective verbs and ditransitives from transitives. | 'cause/make/let' or added causer. | [PDF p. 23] |
| Potential | potential construction with transitive verb form. | 'can/be able to' style meaning. | [PDF p. 24] |
| Light verbs | lexical items combine with -oq- 'become' or -od- 'do' for borrowed/new verbs. | translate as predicate meaning, not literally 'do/become' when lexicalized. | [PDF p. 18], [PDF p. 34] |

## Agreement
| Fact | Translation cue | Source page |
|---|---|---|
| Gender agreement prefixes occur on many vowel-initial adjectives and verbs, some adverbs/postpositions, and one particle. | agreement marker can identify absolutive argument gender/number. | [PDF p. 7] |
| Predicate and some adverbs agree with the absolutive noun phrase regardless of transitivity. | agreement tracks absolutive, not necessarily English subject. | [PDF p. 21] |
| Long-distance agreement can occur with an absolutive argument in an embedded clause. | agreement can point to embedded topic-like absolutive NP. | [PDF p. 21-22] |

## Negation
| Form/fact | Function | Translation cue | Source page |
|---|---|---|
| Sentential negation | negative forms of the verb. | use verb negation for clause-level 'not'. | [PDF p. 31] |
| Constituent negation | particle a:nu follows negated constituent. | translate local 'not X'. | [PDF p. 31] |
| Multiple negation | impossible according to PDF. | avoid interpreting repeated negatives as normal concord. | [PDF p. 31] |
| Negative imperative/prohibitive | forms include negative imperative/prohibitive categories. | command 'do not...' | [PDF p. 16], [PDF p. 40] |

## Questions
| Form/fact | Function | Translation cue | Source page |
|---|---|---|
| -(j)a: | yes-no interrogative suffix added to focused constituent. | focused yes/no question marker. | [PDF p. 25] |
| Content questions | wh-word position depends on whether it is interpreted exhaustively. | wh-words may move or remain in place. | [PDF p. 25] |
| Multiple wh-questions | rare, fixed wh-word order. | preserve/follow fixed order if multiple wh forms appear. | [PDF p. 26] |
| Question verb forms | declarative verb form in all tenses except past witnessed. | do not expect special verb question morphology except noted cases. | [PDF p. 26] |

## Clause structure
| Clause type | Function | Translation cue | Source page |
|---|---|---|
| Intransitive | single absolutive argument. | absolutive NP often English subject. | [PDF p. 22] |
| Ergative/transitive | ergative agent and absolutive patient. | ergative NP often English subject/agent; absolutive NP often object. | [PDF p. 22] |
| Ditransitive | agent ergative, patient absolutive, recipient in lative or locative depending transfer. | recipient may translate as 'to/for'. | [PDF p. 23] |
| Affective clause | experiencer in lative, stimulus in absolutive. | lative experiencer may be English subject. | [PDF p. 24] |
| Biabsolutive construction | occurs with two analytical verbal predicate types. | two absolutives require care in role assignment. | [PDF p. 24] |
| Non-verbal predication | covered as a clause type. | predicate may not be verbal. | [PDF p. 33] |

## Subordinate, relative, complement, and adverbial clauses
| Construction | Function | Translation cue | Source page |
|---|---|---|
| Relative clauses | predicate is a participle; arguments and adjuncts are relativizable. | translate participial modifier as English relative clause. | [PDF p. 27] |
| Correlative clauses | wh-word + interrogative verb form. | translate as 'whoever/whatever/wherever...' when context supports. | [PDF p. 28] |
| Complement clauses | nominalized complements and finite-verb complements are described. | translate as English gerund/that-clause depending construction. | [PDF p. 28] |
| Adverbial clauses | converbs are very common. | converb clauses often translate as temporal, locative, causal, or sequential clauses. | [PDF p. 29] |
| Infinitival clauses | occur with modal, phasal, motion, psychological verbs, and t'amizi -od- 'cause'. | translate as English infinitive/purpose/complement clause. | [PDF p. 30] |
| Clause chaining | rich converb inventory permits free clause chaining. | multiple converbs can link to one finite clause. | [PDF p. 32] |

## Particles, postpositions, and focus
| Form/fact | Function | Translation cue | Source page |
|---|---|---|
| Postpositions | many also function as adverbs. | usually translate as English prepositions/adverbs. | [PDF p. 19-20] |
| a:nu | constituent negation particle after negated constituent. | 'not X'. | [PDF p. 31] |
| -kin / -tow | focus markers. | identify contrast/focus material. | [PDF p. 33] |

## Glossing abbreviation table
{abbrev_table()}

## Important examples from the PDF
| Page | Example/cue |
|---|---|
| [PDF p. 20] | ħon-ƛ'o-si ˁadala 'the fool on the hill'; linker after oblique modifier. |
| [PDF p. 22] | intransitive and ergative examples show absolutive/ergative alignment. |
| [PDF p. 23] | ditransitive and causative examples show recipient/causee case behavior. |
| [PDF p. 24] | affective clause examples show lative experiencer and absolutive stimulus. |
| [PDF p. 25] | yes-no questions use -(j)a: on the focused constituent. |
| [PDF p. 27] | relative clause examples use participial predicates. |
| [PDF p. 29] | temporal, locative, and causal adverbial clauses are illustrated with converbs. |
| [PDF p. 31] | constituent negation is shown with a:nu following the negated constituent. |

## Not found or unclear in this PDF
| Area | Status |
|---|---|
| Articles/definiteness | No article system was identified in the extracted text; demonstratives are relevant. |
| Full pronoun paradigms | Pronoun forms are discussed, but table extraction is noisy; verify exact paradigms in page JPGs. |
| Full postposition list | Examples are given, but the PDF refers elsewhere for fuller coverage. |
| Some table cells | Embedded text order for complex tables is imperfect; selected JPG pages preserve visual layout. |
"""


def cheatsheet_content(pdf_name: str) -> str:
    return f"""
# Tsez Compact Translation Cheat Sheet

Source: {pdf_name}. Use only these PDF-derived cues.

## Core cues
| Label | Cue | Page |
|---|---|---|
| WORD_ORDER | Head-final; postpositions; prenominal relatives/adjectives/genitives/numerals; dependent clauses before independent. | [PDF p. 20] |
| CASE | Morphologically ergative: ABS for intransitive S/transitive O; ERG for transitive A. | [PDF p. 21-23] |
| AGREEMENT | Predicate agrees with absolutive NP in gender/number when agreement is visible. | [PDF p. 21] |
| QUESTION | yes-no question suffix -(j)a: attaches to focused constituent. | [PDF p. 25] |
| NEGATION | sentential negation uses negative verb forms; constituent negation uses a:nu after negated item. | [PDF p. 31] |

## Case and local forms
| Form | Function | Translation cue | Page |
|---|---|---|---|
| -Ø ABS | absolutive | S/O role cue | [PDF p. 9] |
| -a: ERG | ergative | transitive agent | [PDF p. 9] |
| -s GEN1 / -zo GEN2 | genitives | possessor/modifier | [PDF p. 9] |
| -r LAT | lative | to/toward; experiencer in affective clauses | [PDF p. 9], [PDF p. 24] |
| IN/CONT/SUPER/SUB/AD/APUD/POSS | local series | in/among/on/under/at/near/on vertical | [PDF p. 9-10] |
| ESS/LAT/ABL/VERS | local dimensions | at/to/from/towards | [PDF p. 9-10] |

## Verb cues
| Form/fact | Translation cue | Page |
|---|---|---|
| PSTWIT vs PSTUNW | witnessed/unwitnessed past evidentiality | [PDF p. 15], [PDF p. 40] |
| -č'V negative | verbal negation suffix family | [PDF p. 16] |
| participles | relative clauses/modifiers | [PDF p. 16], [PDF p. 27] |
| converbs | adverbial clauses and clause chaining | [PDF p. 17], [PDF p. 29], [PDF p. 32] |
| infinitive -a | infinitival complements | [PDF p. 17], [PDF p. 30] |
| causative -r- | adds causer; in transitive base, causee in poss.essive | [PDF p. 23] |
| potential | ability/potential construction | [PDF p. 24] |

## Clause patterns
| Pattern | Translation cue | Page |
|---|---|---|
| affective: experiencer LAT + stimulus ABS | English experiencer subject may be lative NP. | [PDF p. 24] |
| ditransitive: A ERG + patient ABS + recipient LAT/local | translate recipient as to/for. | [PDF p. 23] |
| relative: participial predicate | English relative clause. | [PDF p. 27] |
| adverbial: converb | when/after/because/while style clause from context. | [PDF p. 29] |

## Mini abbreviations
{abbrev_table()}
"""


def summary_tables_content(pdf_name: str) -> str:
    return f"""
# Tsez Structured Summary with Tables

## Source
| Field | Value |
|---|---|
| PDF | {pdf_name} |
| Use | Qwen-VL visual grammar context |

## Word order
| Cue | Meaning | Page |
|---|---|---|
| head-final | dependent before head; postpositions | [PDF p. 20] |
| prenominal relatives | relative clause before noun | [PDF p. 20], [PDF p. 27] |
| predicate often clause-medial | verb not always final | [PDF p. 20] |

## Nouns and cases
| Form | Function | Translation cue | Page |
|---|---|---|---|
| ABS -Ø | absolutive | intransitive S / transitive O | [PDF p. 9], [PDF p. 21-23] |
| ERG -a: | ergative | transitive A | [PDF p. 9], [PDF p. 21-23] |
| GEN1 -s | genitive 1 | possessor/modifier | [PDF p. 9] |
| GEN2 -zo | genitive 2 | case concord/linked genitive | [PDF p. 9] |
| LAT -r | lative | to/toward; affective experiencer | [PDF p. 9], [PDF p. 24] |
| local cases | localization + dimension | local relations | [PDF p. 9-10] |

## Pronouns and demonstratives
| Area | Cue | Page |
|---|---|---|
| personal pronouns | first/second person only; third person uses demonstratives | [PDF p. 12] |
| demonstratives | gender I vs II-IV in forms | [PDF p. 13] |
| interrogatives | wh-forms; position depends on exhaustivity | [PDF p. 13], [PDF p. 25] |

## Verbs
| Area | Cue | Page |
|---|---|---|
| TAM/evidentiality | tense-mood-aspect-evidentiality obligatory except noted forms | [PDF p. 15] |
| imperative | zero for intransitive/derived transitive; -o for simple transitive | [PDF p. 16] |
| negation | -č'V suffix family; negative forms may be irregular | [PDF p. 16] |
| non-finites | participles, converbs, infinitive, verbal noun | [PDF p. 16-17] |
| causative | -r- adds causer | [PDF p. 23] |
| light verbs | -oq- 'become', -od- 'do' with lexical items | [PDF p. 18], [PDF p. 34] |

## Negation and questions
| Form | Function | Page |
|---|---|---|
| -(j)a: | yes-no question suffix on focused constituent | [PDF p. 25] |
| a:nu | constituent negation after negated constituent | [PDF p. 31] |
| negative verb forms | sentential negation | [PDF p. 31] |
| multiple negation impossible | do not interpret as concord | [PDF p. 31] |

## Clause patterns
| Clause | Cue | Page |
|---|---|---|
| ergative | A=ERG, O=ABS | [PDF p. 21-23] |
| agreement | predicate agrees with absolutive NP | [PDF p. 21] |
| affective | experiencer LAT, stimulus ABS | [PDF p. 24] |
| relative | participial predicate | [PDF p. 27] |
| complement | nominalized or finite complements | [PDF p. 28] |
| adverbial | converb clauses common | [PDF p. 29] |
| infinitival | modal/phasal/motion/psych verbs | [PDF p. 30] |

## Gloss abbreviations
{abbrev_table()}
"""


def write_selected_manifest(pdf_path: Path, texts: list[str], out_dir: Path) -> None:
    md_lines = ["# Selected Impactful PDF Pages", ""]
    txt_lines = ["Selected Impactful PDF Pages", ""]
    for page_number in SELECTED_PAGES:
        text = texts[page_number - 1] if page_number - 1 < len(texts) else ""
        score, reasons = score_page(text)
        filename = f"page_{page_number:03d}.jpg"
        info = first_lines(text, 10)[:500]
        reason = selected_reason(page_number)
        use = usefulness(page_number)
        concerns = "Embedded text extracted; verify exact symbols and table alignment in JPG."

        md_block = [
            f"## PDF page {page_number}",
            f"- JPG filename: {filename}",
            f"- Selection score: {score}",
            f"- Reason for selection: {reason}; {'; '.join(reasons) if reasons else 'manual coverage'}",
            f"- Grammar information on the page: {info}",
            f"- Expected usefulness for translation: {use}",
            f"- Readability or OCR concerns: {concerns}",
            "",
        ]
        txt_block = [
            f"PDF page {page_number}",
            f"- JPG filename: {filename}",
            f"- Selection score: {score}",
            f"- Reason for selection: {reason}; {'; '.join(reasons) if reasons else 'manual coverage'}",
            f"- Grammar information on the page: {info}",
            f"- Expected usefulness for translation: {use}",
            f"- Readability or OCR concerns: {concerns}",
            "",
        ]
        md_lines.extend(md_block)
        txt_lines.extend(txt_block)

    (out_dir / "selected_pages_manifest.md").write_text("\n".join(md_lines), encoding="utf-8")
    (out_dir / "selected_pages_manifest.txt").write_text("\n".join(txt_lines), encoding="utf-8")


def optional_notes(pdf_path: Path, texts: list[str], out_dir: Path, rendered: list[str]) -> str:
    skipped = [str(i) for i in range(1, len(texts) + 1) if i not in SELECTED_PAGES]
    notes = f"""
# Optional Page Notes

## Source
- PDF: {pdf_path.name}
- Embedded text extraction: available for all pages checked.
- OCR: not used.

## Selected pages
- Rendered selected page JPGs: {', '.join(rendered)}
- The selected pages prioritize the highest-impact original PDF images for GPU-safe VLM runs: noun class/gender, case paradigms, local cases, pronouns, verb TAM/negation/non-finites, word order, agreement, and questions.
- Previously selected but now skipped pages covered overlapping demonstrative details, converbs/relatives/adverbials, ditransitives, constituent negation, and abbreviation lists. Those facts remain represented in the Markdown/text summaries and rendered summary images.

## Pages skipped
- Skipped pages: {', '.join(skipped)}
- Reason: lower direct usefulness for translation, bibliography/reference material, phonological background, sociolinguistic background, or overlap with selected grammar pages.

## Uncertainty and rendering issues
- Some table extraction from PDF text is linearized and imperfect; the selected JPG pages preserve original visual layout.
- Unicode symbols such as ɬ, ƛ, χ, ˁ, ʁ, č, and pharyngealization symbols were preserved in Markdown/text and rendered with DejaVu fonts.
- No external linguistic information was added.
- If a model needs exact table cells, use the selected page JPGs alongside the Markdown summary.
"""
    (out_dir / "optional_page_notes.md").write_text(notes.strip() + "\n", encoding="utf-8")
    (out_dir / "optional_page_notes.txt").write_text(re.sub(r"^#+\s*", "", notes.strip(), flags=re.M) + "\n", encoding="utf-8")
    return notes


def process_pdf(pdf_path: Path) -> None:
    pdf_name = pdf_path.stem
    out_root = MATERIALS_ROOT / pdf_name
    summary_dir = out_root / "Summary Text"
    cheatsheet_dir = out_root / "Cheat Sheet from PDF"
    tables_dir = out_root / "Summary Image with Tables"
    pages_dir = out_root / "Pages from the PDF"
    selected_dir = pages_dir / "Selected impactful pages JPG"

    texts = page_texts(pdf_path)

    write_both(
        summary_dir / "summary_model_readable.md",
        summary_dir / "summary_model_readable.txt",
        summary_content(pdf_path.name),
    )
    write_both(
        cheatsheet_dir / "compact_cheatsheet.md",
        cheatsheet_dir / "compact_cheatsheet.txt",
        cheatsheet_content(pdf_path.name),
    )
    write_both(
        tables_dir / "summary_tables.md",
        tables_dir / "summary_tables.txt",
        summary_tables_content(pdf_path.name),
    )

    rendered_page_names = []
    selected_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {f"page_{page_number:03d}.jpg" for page_number in SELECTED_PAGES}
    for old_jpg in selected_dir.glob("page_*.jpg"):
        if old_jpg.name not in keep_names:
            old_jpg.unlink()
    for page_number in SELECTED_PAGES:
        jpg = selected_dir / f"page_{page_number:03d}.jpg"
        render_pdf_page(pdf_path, page_number, jpg, zoom=2.0)
        rendered_page_names.append(jpg.name)

    write_selected_manifest(pdf_path, texts, selected_dir)
    optional_notes(pdf_path, texts, pages_dir, rendered_page_names)

    cheat_images = render_markdown_to_jpg(cheatsheet_dir / "compact_cheatsheet.md", cheatsheet_dir / "compact_cheatsheet.jpg")
    table_images = render_markdown_to_jpg(tables_dir / "summary_tables.md", tables_dir / "summary_tables.jpg")

    report = f"""
LANGUAGE: Tsez
PDF file processed: {pdf_path}
Output location: {out_root}

Files created:
- Summary Text/summary_model_readable.md
- Summary Text/summary_model_readable.txt
- Cheat Sheet from PDF/compact_cheatsheet.md
- Cheat Sheet from PDF/compact_cheatsheet.txt
- Cheat Sheet from PDF/{', '.join(p.name for p in cheat_images)}
- Summary Image with Tables/summary_tables.md
- Summary Image with Tables/summary_tables.txt
- Summary Image with Tables/{', '.join(p.name for p in table_images)}
- Pages from the PDF/Selected impactful pages JPG/*.jpg
- Pages from the PDF/Selected impactful pages JPG/selected_pages_manifest.md
- Pages from the PDF/Selected impactful pages JPG/selected_pages_manifest.txt
- Pages from the PDF/optional_page_notes.md
- Pages from the PDF/optional_page_notes.txt

Number of selected original PDF pages rendered to JPG: {len(SELECTED_PAGES)}
Selected impactful pages: {', '.join(str(p) for p in SELECTED_PAGES)}

Major grammar areas covered:
- gender/noun class and agreement
- number and case
- local cases and spatial nouns
- pronouns and demonstratives
- verb classes, TAM/evidentiality, imperative, negation, non-finites
- causative, potential, light verbs
- head-final word order and noun phrase structure
- ergative alignment, affective clauses, ditransitives
- questions, relative clauses, complement clauses, adverbial/converb clauses
- constituent and sentential negation
- glossing abbreviations

Missing or unclear information:
- Full pronoun and demonstrative tables are partly noisy in extracted text; source JPGs preserve visual layout.
- Some table cells are linearized by PDF extraction.
- Articles/definiteness were not identified as an available section in this PDF.

OCR, extraction, font, or JPG rendering issues:
- OCR was not used because embedded PDF text was available.
- Unicode symbols were preserved and rendered with DejaVu fonts.
- Rendered Markdown JPGs were created from the authoritative Markdown sources.

Consistency and faithfulness:
- Materials are based only on the input PDF.
- No external linguistic information was added.
- Markdown/text and rendered JPG image versions were created successfully.
- Selected page JPGs cover the most compact high-impact original PDF page set for GPU-safe Qwen-VL runs.

Ready for Qwen/Qwen-VL translation experiments: yes.
"""
    (out_root / "creation_report.txt").write_text(report.strip() + "\n", encoding="utf-8")
    print(report.strip())


def main() -> None:
    pdfs = find_pdfs(PDF_PATH_OR_FOLDER)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {PDF_PATH_OR_FOLDER}")
    for pdf in pdfs:
        process_pdf(pdf)


if __name__ == "__main__":
    main()
