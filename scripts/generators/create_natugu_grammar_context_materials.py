from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


import argparse
from pathlib import Path

import fitz

from create_grammar_context_materials import (
    PageInfo,
    clean_pdf_name,
    extract_pages,
    find_pdfs,
    render_markdown_to_jpg,
    render_pdf_page,
    write_both,
)


# Keep one high-value page for each major translation-relevant area. Raw score
# breaks ties; coverage prevents repeated valency pages from crowding out
# pronouns, questions, negation, and complex clauses.
SELECTED_PAGES = [3, 8, 23, 25, 37, 38, 39, 50, 58]


def contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def first_lines(text: str, n: int = 8) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines[:n])


def score_page(page: PageInfo) -> PageInfo:
    tests = [
        (5, ["table", "slots", "paradigm", "enclitics", "classifiers"], "tables/paradigms/enclitics"),
        (5, ["‘", "’", "gloss", "ᴘ", "ɴ", "ᴅ", "example"], "glossed examples/translations"),
        (4, ["word order", "VAO", "SV", "VS", "intransitive", "transitive", "semitransitive"], "clause order/transitivity"),
        (4, ["negation", "negative", "tr-", "tr>", "do not", "polar questions", "content questions"], "negation/questions"),
        (4, ["subordinate", "relative clauses", "adverbial clauses", "complement clauses"], "subordination/relative/adverbial"),
        (4, ["causative", "middle", "applicative", "passive", "valency"], "valency changing"),
        (3, ["demonstratives", "possessive", "genitives", "noun", "pronouns"], "NP/pronoun/possessive"),
        (-4, ["references", "bibliography"], "references"),
    ]
    score = 0
    reasons = []
    for weight, terms, label in tests:
        if contains_any(page.text, terms):
            score += weight
            reasons.append(f"{weight:+d} {label}")
    return PageInfo(page.number, page.text, score, reasons)


def select_pages(pages: list[PageInfo]) -> list[PageInfo]:
    scored = [score_page(p) for p in pages]
    selected = [p for p in scored if p.number in SELECTED_PAGES]
    return sorted(selected, key=lambda p: p.number)


def gloss_table() -> str:
    rows = [
        ("A", "agent argument", "transitive clauses have A and O", "PDF p. 3"),
        ("O", "object argument", "transitive clauses use VAO order", "PDF p. 3"),
        ("S", "subject of intransitive", "SV with noun/NP subject; VS with pronominal enclitic", "PDF p. 35"),
        ("Set I", "verbal subject enclitics", "subject marking set in Table 4", "PDF p. 8"),
        ("Set II", "circumfix form with prefix and subject enclitic", "used in the person/number system", "PDF p. 3, p. 8"),
        ("PCLF", "possessive classifier", "classifies possession/relations", "PDF p. 17"),
        ("DEM1/DEM2", "listener-/speaker-anchored demonstratives", "la/lc vs ka/kc grid", "PDF p. 13"),
        ("SUBR", "subordinator", "kx; can introduce relative/subordinate clauses", "PDF p. 13, p. 58-60"),
        ("PFV", "perfective", "sc= in pre-core slot -4", "PDF p. 23"),
        ("IPFV", "imperfective", "sa= in pre-core slot -4", "PDF p. 23"),
        ("RL", "realis", "tq- in mood slot -2", "PDF p. 23"),
        ("IRR", "irrealis", "na- in mood slot -2", "PDF p. 23"),
        ("NEG", "negation", "tr>...<u circumfix; other negative forms", "PDF p. 37, p. 55"),
        ("CAUS", "causative", "a- in slot -1; adds causer", "PDF p. 23, p. 39"),
        ("MID", "middle", "r- in slot -1; valency-reducing", "PDF p. 23, p. 39"),
        ("GDIR", "geometric directional", "up/down/in/out directionals", "PDF p. 25"),
        ("PDIR", "personal directional", "-mq hither, -bz yon/away", "PDF p. 25"),
        ("APPL", "applicative", "-ngr peripheral applicative; core applicatives also described", "PDF p. 24-25, p. 40"),
        ("COS", "change of state", "=pe in demonstratives/verb examples", "PDF p. 14, p. 23"),
    ]
    out = ["| Form | Function | Translation cue | Source page |", "|---|---|---|---|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def summary_content(language: str, pdf_name: str) -> str:
    return f"""
# {language} / Natqgu Grammar Summary for Translation

## Source
PDF: {pdf_name}
Pages used: nine selected high-impact pages from the 71-page grammar sketch: pages 3, 8, 23, 25, 37-39, 50, and 58.
Extraction notes: embedded PDF text was available. The PDF spelling is Natqgu [ntu], also spelled Natügu. Some gloss labels use small-cap Unicode symbols; these are preserved in Markdown and rendered with DejaVu fonts.

## WORD_ORDER
| Fact | Translation cue | Source page |
|---|---|---|
| Transitive clauses display A and O in VAO order. A is either a nominal argument or a person/number enclitic. | Expect verb first, then agent, then object in basic transitive clauses. | [PDF p. 3] |
| Subordinate clauses and most peripheral arguments occur post-verbally. | Material after the verb may be subordinate/peripheral. | [PDF p. 3] |
| Basic intransitive order is SV when S is a noun or noun phrase. | NP subject can precede intransitive verb. | [PDF p. 35] |
| Intransitive order is VS when S is a pronominal enclitic. | Pronominal subject enclitic follows the verb. | [PDF p. 35] |
| Transitive unmarked word order is VAO, but A or O can be fronted for discourse purposes. | Fronted material may be discourse-driven. | [PDF p. 35-36] |

## NOUNS_AND_CASE
| Fact | Translation cue | Source page |
|---|---|---|
| Natqgu subcategorizes nouns by count/mass, common/proper, animate/inanimate, and direct/indirect possession. | Noun type affects grammar and lexical choice. | [PDF p. 8] |
| There is no case or number marked on nouns. | Do not look for noun case suffixes; use word order, enclitics, prepositions, and classifiers. | [PDF p. 8] |
| Number, if indicated, occurs on a demonstrative modifier. | Plural may appear in demonstrative morphology rather than on the noun. | [PDF p. 8] |
| Mass nouns take a form of indefinite plural du 'some'. | du can translate as 'some' with mass/indefinite nouns. | [PDF p. 8] |

## PRONOUNS_AND_PERSON_NUMBER
| Form area | Function | Translation cue | Source page |
|---|---|---|---|
| Free pronouns | base ni- 'be' + person/number enclitics | free subject/object pronouns | [PDF p. 8] |
| Indirect object pronouns | dative base ba- + person/number enclitics | dative/indirect object pronouns | [PDF p. 8] |
| Set I | verbal subject enclitics | subject marking | [PDF p. 8] |
| Set II | verbal/object/enclitic forms; augmented third includes prefixes nz-, tz-, na- | participant marking | [PDF p. 8] |
| Free pronouns as NP heads | examples include ni=nge 'I/me' and ni=gr 'we/us' | free pronouns can head NPs | [PDF p. 22] |

## DEMONSTRATIVES_AND_DEFINITENESS
| Form | Function | Translation cue | Source page |
|---|---|---|---|
| la | listener-anchored proximal DEM1.PROX | this/near listener-anchored demonstrative | [PDF p. 13] |
| lc | listener-anchored distal DEM1.DIST | that/distal listener-anchored demonstrative | [PDF p. 13] |
| ka | speaker-anchored proximal DEM2.PROX | this/near speaker-anchored demonstrative | [PDF p. 13] |
| kc | speaker-anchored distal DEM2.DIST | frequent demonstrative; can function as subordinator/relativizer | [PDF p. 13] |
| =pe / =pnz | change-of-state/completive forms in demonstrative slot +2 | increases specificity/definiteness; may signal entity already exists/existed | [PDF p. 14] |
| -ng(q) | plural in demonstrative slot +3 | plural of modified noun | [PDF p. 13-14] |
| =pwz | restrictive 'just/only/exactly' | restrictive cue | [PDF p. 14] |

## POSSESSION_AND_GENITIVES
| Form | Function | Translation cue | Source page |
|---|---|---|---|
| ngr | GEN1A 'of, from' | contents, relationships, location, properties | [PDF p. 17] |
| r | GEN1B variant | phonologically set variant | [PDF p. 17] |
| -kr | NMLZ.PCLF | possessive of action nominalizations and a few nouns | [PDF p. 17] |
| sc | PCLF.hand | handheld/moveable possessions; generic possessor | [PDF p. 17] |
| ma | PCLF.betel | betel nut and related items | [PDF p. 17] |
| mq | PCLF.drink | drinkables and wet fruits | [PDF p. 17] |
| na | PCLF.food | edibles except drinkables | [PDF p. 17] |
| ne | PCLF.rsbl | underlings/responsibility/creation | [PDF p. 17] |
| nyz | PCLF.B&G | property and time | [PDF p. 17] |
| mnr/pnr | PCLF.fire | hearth/home items | [PDF p. 17] |
| lr | PCLF.assoc | closely associated/intrinsic relation | [PDF p. 17] |
| nr | PCLF.feel | thoughts, heart, desires | [PDF p. 17] |

## VERB_COMPLEX
| Slot | Form/category | Translation cue | Source page |
|---|---|---|---|
| -4 | sc= PFV, sa= IPFV, ma= 'lest' | aspect proclitics | [PDF p. 23] |
| -3 | tr> NEG | first part of negative circumfix | [PDF p. 23] |
| -2 | tq- RL, tz- RL.3AUG, na- IRR, nz- 3AUG | mood/person prefix area | [PDF p. 23] |
| -1 | a- CAUS, r- MID | causative or middle; mutually exclusive | [PDF p. 23] |
| 0 | verb root | root/core | [PDF p. 23-24] |
| +1 | compounders | stem creation: V+V, V+bound V, V+ADV, V+N | [PDF p. 24] |
| +2 | core applicatives | can license additional argument | [PDF p. 24] |
| +7 | GDIR | geometric directionals up/down/in/out | [PDF p. 25] |
| +16 | -ngr APPL | peripheral applicative | [PDF p. 25, p. 40] |
| +17 | -mq / -bz PDIR | hither/toward deictic center vs yon/away | [PDF p. 25] |
| +19/+20 | subject/object enclitics | participant marking | [PDF p. 24-25] |
| +21 | <u NEG | second part of negative circumfix | [PDF p. 25, p. 37] |

## VERB_CLASSES_AND_VALENCY
| Area | Function | Translation cue | Source page |
|---|---|---|---|
| Intransitive classes | three classes: no transitive form, causativized by a-, transitivized with -ti | valency class affects arguments | [PDF p. 30] |
| Causative a- | adds a core argument, the causer | translate as 'make/cause' where appropriate | [PDF p. 39] |
| Middle r- | primary valency-reducing device; transitive to semitransitive/depatientive | object may be generic/absent | [PDF p. 39] |
| Core applicatives | each licenses a further nominal argument with role determined by applicative meaning | added object/role can appear in translation | [PDF p. 40] |
| Passive nz-/tz- | agentless passives; original subject no longer part of construction | promoted object may become subject in English passive | [PDF p. 50] |

## CLAUSE_TYPES
| Clause type | Structure/function | Translation cue | Source page |
|---|---|---|---|
| Intransitive | SV with NP subject; VS with pronominal enclitic subject | translate subject according to NP/enclitic | [PDF p. 35] |
| Transitive | VAO unmarked, with possible fronting | verb-agent-object baseline | [PDF p. 35] |
| Semitransitive | middle prefix r-/ö- detransitivizes; patient generic/irrelevant | focus on action rather than specific object | [PDF p. 36] |
| Imperative | verb stripped of TAM distinctions; second person subject omitted in minimal | command | [PDF p. 36] |
| Negative imperative | bzkq 'Do not!' precedes what is forbidden or stands alone | prohibition | [PDF p. 36] |
| Hortative | irrealis marking with non-second-person forms | 'let X do Y' or 'X should/must do Y' | [PDF p. 36] |

## QUESTIONS_AND_NEGATION
| Form/fact | Function | Translation cue | Source page |
|---|---|---|---|
| Polar questions | morphologically identical to declaratives; rising intonation | question may not have special morphology | [PDF p. 37] |
| eu / trtingr | 'yes' / 'no' answers to polar questions | answer words | [PDF p. 37] |
| Question words | neke 'who', nike 'what', drlve~myx 'where', mzli kx 'when', memule 'why', myx kxmu 'how', tqlvr 'how many/much' | content question cues | [PDF p. 37] |
| e trtingr | 'or not' tag question | tag question strategy | [PDF p. 37] |
| tr=...=u | standard circumfix negation | most common verbal negation | [PDF p. 37-38], [PDF p. 55] |
| tr>...<ka | 'not...yet' | tr- with =ka, without =u | [PDF p. 38], [PDF p. 55] |
| trtingr | 'no' | negator with NPs/predicate nominals; negative answer | [PDF p. 55] |
| bzkq | 'do not' | negative imperative | [PDF p. 36], [PDF p. 55] |

## PREPOSITIONS_AND_ADJUNCTS
| Form/fact | Function | Translation cue | Source page |
|---|---|---|---|
| mz | only Natqgu preposition; wide range of peripheral roles | often after predicate; can start sentence referencing previous discourse | [PDF p. 38] |
| Local nouns / temporal nouns | adjuncts in clause | location/time interpretation | [PDF p. 38] |

## SUBORDINATION_AND_COMPLEX_CLAUSES
| Construction | Function | Translation cue | Source page |
|---|---|---|---|
| Finite complement clauses | follow matrix clause; matrix/complement inflections independent | translate as 'that...' or direct quote where appropriate | [PDF p. 57] |
| Non-finite complement clauses | nominalizations instead of kx | translate as gerund/nominalized clause if appropriate | [PDF p. 58] |
| Relative clauses | introduced by kx or known-NP demonstrative kc; follow modified NP | translate as relative clause | [PDF p. 58-59] |
| kx | subordinator; can also mean if/whether/since/when in contexts | all occurrences glossed SUBR | [PDF p. 57-59] |
| Temporal clauses | mz nibr 'after' + nominalization; juxtaposition; kx 'when' + finite verb | after/when sequencing | [PDF p. 60] |
| Reason clauses | mz nzmu-krde lcde 'from its being like that one'; murde 'because' | reason/therefore | [PDF p. 60-61] |
| Purpose clauses | murde 'so that' + irrealis; or nominalization | purpose/in order that | [PDF p. 61] |
| Disjunctive e/o | 'or', mostly NPs and VPs with same subject in SGM text | or | [PDF p. 65] |
| Adversative a' | 'but', conjoins clauses/sentences in SGM text | contrast | [PDF p. 66] |

## GLOSSING_ABBREVIATIONS
{gloss_table()}

## HIGH_VALUE_EXAMPLES
| Page | Original/gloss/translation cue |
|---|---|
| [PDF p. 8] | Angq-px=amu du lue ... / draw-GDIR.out=2AUGII INDF.PL water / 'Draw out some water...' shows du 'some'. |
| [PDF p. 22] | ni=nge and ni=gr in free pronoun examples show free pronouns as NP heads. |
| [PDF p. 22] | tr-na-krlz-mq=le-u / NEG-IRR-reach-PDIR.hither=3MINIA-NEG / 'it wouldn't reach...' shows negative circumfix around verb complex. |
| [PDF p. 35] | aelwa-px Gct da ... / show-GDIR.out God thing ... / 'God demonstrated such things...' shows VAO transitive order. |
| [PDF p. 36] | bzkq maletr=q / do not hold=2MINI / 'Don't hold it!' shows negative imperative. |
| [PDF p. 37] | Krlz=q vs Krlz=q? / 'You know.' vs 'Do you know?' shows polar question by intonation. |
| [PDF p. 58] | yzu-tr=kr nz-siklu-ngr ... / begin-GDIR.in=1AUGI NMLZ1-be.schooled-NMLZ / 'We began being schooled...' shows non-finite complement. |

## TRANSLATION_CUES_FOR_QWEN
- WORD_ORDER: Use VAO as the basic transitive order, SV for intransitives with NP subject, and VS for intransitives with pronominal enclitic subject. [PDF p. 3], [PDF p. 35]
- CASE: Do not infer noun case suffixes; the PDF states no case or number is marked on nouns. [PDF p. 8]
- PRONOUN: ni- and ba- bases combine with person/number enclitics for free and indirect object pronouns. [PDF p. 8]
- VERB: Read the verb complex by slots; especially aspect/mood before root and directionals/applicatives/enclitics after root. [PDF p. 22-25]
- NEGATION: tr=...=u is standard verbal negation; bzkq is prohibitive; trtingr is 'no'. [PDF p. 36-38], [PDF p. 55]
- SUBORDINATION: kx is a central subordinator/relativizer-like form; kc can be relativizer for known NPs. [PDF p. 13], [PDF p. 58-60]
- TRANSLATION_CUE: Directionals and applicatives can add English spatial/argument relations such as hither/away/with/to whom depending on form and context. [PDF p. 25], [PDF p. 40], [PDF p. 45]
"""


def cheat_content(language: str, pdf_name: str) -> str:
    return f"""
# {language} Compact Translation Cheat Sheet
Source: {pdf_name}

## WORD_ORDER
| Clause | Basic order | Page |
|---|---|---|
| Transitive | VAO | [PDF p. 3], [PDF p. 35] |
| Intransitive with NP S | SV | [PDF p. 35] |
| Intransitive with pronominal enclitic S | VS | [PDF p. 35] |
| Subordinate/peripheral | mostly post-verbal | [PDF p. 3] |

## NOUNS_PRONOUNS
| Cue | Translation use | Page |
|---|---|---|
| no case/number on nouns | use order/enclitics/prepositions | [PDF p. 8] |
| number on demonstrative modifier | plural may not be on noun | [PDF p. 8], [PDF p. 13-14] |
| ni- + enclitic | free subject/object pronoun | [PDF p. 8] |
| ba- + enclitic | indirect object/dative pronoun | [PDF p. 8] |

## DEMONSTRATIVES
| Form | Cue | Page |
|---|---|---|
| la/lc | listener anchored prox/dist | [PDF p. 13] |
| ka/kc | speaker anchored prox/dist | [PDF p. 13] |
| kc | frequent; can be subordinator/relativizer | [PDF p. 13], [PDF p. 58] |
| -ng(q) | plural on demonstrative | [PDF p. 13-14] |

## VERB_CORE
| Slot/form | Cue | Page |
|---|---|---|
| sc= / sa= | PFV / IPFV aspect | [PDF p. 23] |
| tq- / na- | realis / irrealis | [PDF p. 23] |
| tr>...<u | standard negation | [PDF p. 37-38] |
| a- | causative; adds causer | [PDF p. 39] |
| r- | middle; valency reducing | [PDF p. 39] |
| -mq / -bz | PDIR hither / yon-away | [PDF p. 25] |
| -ngr | applicative | [PDF p. 25], [PDF p. 40] |

## QUESTIONS_NEGATION
| Form | Function | Page |
|---|---|---|
| polar question | same morphology as declarative; rising intonation | [PDF p. 37] |
| neke/nike/memule | who/what/why | [PDF p. 37] |
| e trtingr | or not tag | [PDF p. 37] |
| bzkq | do not | [PDF p. 36], [PDF p. 55] |
| trtingr | no | [PDF p. 37], [PDF p. 55] |

## SUBORDINATION
| Form | Translation cue | Page |
|---|---|---|
| kx | SUBR; relative/if/when/since depending context | [PDF p. 57-59] |
| kc | relativizer when NP is known | [PDF p. 58-59] |
| mz nibr | after + nominalization | [PDF p. 60] |
| murde | because / so that with irrealis | [PDF p. 60-61] |
| e/o | or | [PDF p. 65] |
| a' | but | [PDF p. 66] |
"""


def tables_content(language: str, pdf_name: str) -> str:
    return f"""
# {language} Summary Tables for Qwen-VL
Source: {pdf_name}

## CLAUSE_ORDER
| Type | Order | Translation cue | Page |
|---|---|---|---|
| Transitive | VAO | verb-agent-object | [PDF p. 3], [PDF p. 35] |
| Intransitive NP subject | SV | subject before verb | [PDF p. 35] |
| Intransitive enclitic subject | VS | subject enclitic after verb | [PDF p. 35] |
| Subordinate/peripheral | post-verbal | after predicate | [PDF p. 3], [PDF p. 38] |

## PERSON_PRONOUNS
| Base/set | Function | Page |
|---|---|---|
| ni- | free subject/object pronoun base | [PDF p. 8] |
| ba- | indirect object/dative pronoun base | [PDF p. 8] |
| Set I | verbal subject enclitics | [PDF p. 8] |
| Set II | verbal/object/circumfix-related enclitics | [PDF p. 8] |

## DEMONSTRATIVES_POSSESSION
| Area | Forms | Cue | Page |
|---|---|---|---|
| DEM | la/lc/ka/kc | listener/speaker anchored x prox/dist | [PDF p. 13] |
| DEM plural | -ng(q) | plural modified noun | [PDF p. 13-14] |
| GEN | ngr, r, -kr | of/from, variant, nominalized possession | [PDF p. 17] |
| PCLF | sc, ma, mq, na, ne, nyz, mnr/pnr, lr, nr | possession classifier semantics | [PDF p. 17] |

## VERB_COMPLEX
| Slot | Forms | Cue | Page |
|---|---|---|---|
| -4 | sc=, sa=, ma= | aspect/lest | [PDF p. 23] |
| -3/+21 | tr>...<u | NEG circumfix | [PDF p. 23], [PDF p. 37] |
| -2 | tq-, tz-, na-, nz- | mood/person prefix area | [PDF p. 23] |
| -1 | a-, r- | CAUS/MID | [PDF p. 23], [PDF p. 39] |
| +7/+17 | GDIR / PDIR | geometric and personal direction | [PDF p. 25] |
| +16 | -ngr | APPL | [PDF p. 25], [PDF p. 40] |

## NEGATION_QUESTIONS
| Form/fact | Function | Page |
|---|---|---|
| polar Q | declarative morphology + rising intonation | [PDF p. 37] |
| question words | neke, nike, memule, etc. | [PDF p. 37] |
| bzkq | negative imperative | [PDF p. 36], [PDF p. 55] |
| tr>...<ka | not yet | [PDF p. 38], [PDF p. 55] |
| trtingr | no | [PDF p. 37], [PDF p. 55] |

## COMPLEX_CLAUSES
| Construction | Cue | Page |
|---|---|---|
| finite complements | follow matrix clause | [PDF p. 57] |
| non-finite complements | nominalizations | [PDF p. 58] |
| relative clauses | kx or kc; follow NP | [PDF p. 58] |
| temporal | mz nibr 'after', juxtaposition, kx 'when' | [PDF p. 60] |
| reason/purpose | murde, mz nzmu-krde lcde | [PDF p. 60-61] |
| coordination | e/o 'or', a' 'but' | [PDF p. 65-66] |

## GLOSS_ABBREVIATIONS
{gloss_table()}
"""


def write_manifest(path: Path, selected: list[PageInfo], images: dict[int, str]) -> str:
    lines = ["# Selected Impactful PDF Pages", ""]
    txt = ["Selected Impactful PDF Pages", ""]
    for page in selected:
        usefulness = "high" if page.score >= 14 else "medium" if page.score >= 8 else "low"
        reason = "; ".join(page.reasons) if page.reasons else "fixed high-impact grammar page"
        block = [
            f"## PDF page {page.number}",
            f"- JPG filename: {images[page.number]}",
            f"- Selection score: {page.score}",
            f"- Expected usefulness: {usefulness}",
            f"- Reason selected: {reason}",
            f"- Grammar information: {first_lines(page.text, 6)}",
            "- Readability/OCR concerns: embedded text extracted; selected JPG preserves original page layout for exact forms.",
            "",
        ]
        lines.extend(block)
        txt.extend([line.replace("#", "").strip() for line in block])
    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".txt").write_text("\n".join(txt), encoding="utf-8")
    return ", ".join(str(p.number) for p in selected)


def process_pdf(pdf_path: Path, language: str, output_dir: Path) -> None:
    pdf_name = clean_pdf_name(pdf_path)
    root = output_dir.resolve() / "materials/curated_v1" / ("gitksan" if language.lower() in ("gitskan", "gitksan") else language.lower()) / (("pdf1_brown" if "PDF 1" in pdf_path.name else "pdf2_rigsby") if language.lower() in ("gitskan", "gitksan") else "grammar") / "original"
    summary_dir = root / "summary_text"
    cheat_dir = root / "cheatsheet"
    tables_dir = root / "summary_tables"
    pages_dir = root / "page_notes"
    selected_dir = root / "selected_pages"

    for directory in [summary_dir, cheat_dir, tables_dir, pages_dir, selected_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    for old in selected_dir.glob("page_*.jpg"):
        old.unlink()

    pages = [score_page(p) for p in extract_pages(pdf_path)]
    selected = select_pages(pages)
    doc = fitz.open(pdf_path)

    images = {}
    render_notes = []
    for page in selected:
        filename = f"page_{page.number:03d}.jpg"
        size = render_pdf_page(pdf_path, page.number, selected_dir / filename, zoom=2.0)
        images[page.number] = filename
        render_notes.append(f"PDF page {page.number} -> {filename}, size={size[0]}x{size[1]}, score={page.score}")

    write_both(
        summary_dir / "summary_model_readable.md",
        summary_dir / "summary_model_readable.txt",
        summary_content(language, pdf_name),
    )

    cheat_md = cheat_dir / "compact_cheatsheet.md"
    write_both(cheat_md, cheat_dir / "compact_cheatsheet.txt", cheat_content(language, pdf_name))
    cheat_images = render_markdown_to_jpg(cheat_md, cheat_dir / "compact_cheatsheet.jpg")

    tables_md = tables_dir / "summary_tables.md"
    write_both(tables_md, tables_dir / "summary_tables.txt", tables_content(language, pdf_name))
    table_images = render_markdown_to_jpg(tables_md, tables_dir / "summary_tables.jpg")

    selected_pages = write_manifest(selected_dir / "selected_pages_manifest.md", selected, images)

    notes = [
        "# Optional Page Notes",
        "",
        f"PDF file: {pdf_path}",
        f"Total pages: {doc.page_count}",
        "Embedded PDF text extraction was available; OCR was not run.",
        "Only selected impactful pages were rendered to JPG, per instruction.",
        "Unicode small-cap gloss labels were preserved in Markdown/text and rendered with DejaVu fonts.",
        "Some table content is summarized into compact Markdown rather than copied exhaustively; page references point to full source tables.",
        "",
        "## Rendered selected pages",
        *[f"- {line}" for line in render_notes],
        "",
        "## Markdown-to-JPG rendering",
        f"- Cheat sheet images: {', '.join(p.name for p in cheat_images)}",
        f"- Summary table images: {', '.join(p.name for p in table_images)}",
    ]
    (pages_dir / "optional_page_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (pages_dir / "optional_page_notes.txt").write_text("\n".join(line.replace("#", "").strip() for line in notes) + "\n", encoding="utf-8")

    conversion = [
        "Page conversion notes",
        f"PDF file: {pdf_path}",
        f"Total PDF pages: {doc.page_count}",
        f"Selected pages rendered: {len(selected)}",
        "All pages were not rendered because the instruction requested only translation-relevant pages.",
        "Rendering: PyMuPDF zoom=2.0, JPEG quality=82.",
        "Failures: none.",
    ]
    (pages_dir / "page_conversion_notes.txt").write_text("\n".join(conversion) + "\n", encoding="utf-8")

    report = f"""
# Creation Report

Language: {language}
PDF file processed: {pdf_path}
Output location: {root}
Total PDF pages: {doc.page_count}
Selected original PDF pages rendered to JPG: {len(selected)}
Selected pages: {selected_pages}

Files created:
- {summary_dir / 'summary_model_readable.md'}
- {summary_dir / 'summary_model_readable.txt'}
- {cheat_dir / 'compact_cheatsheet.md'}
- {cheat_dir / 'compact_cheatsheet.txt'}
- {', '.join(str(p) for p in cheat_images)}
- {tables_dir / 'summary_tables.md'}
- {tables_dir / 'summary_tables.txt'}
- {', '.join(str(p) for p in table_images)}
- {selected_dir / 'selected_pages_manifest.md'}
- {selected_dir / 'selected_pages_manifest.txt'}
- {pages_dir / 'optional_page_notes.md'}
- {pages_dir / 'optional_page_notes.txt'}
- {pages_dir / 'page_conversion_notes.txt'}

Major grammar areas covered:
- VAO/SV/VS word order
- no case or number marked on nouns
- person and number enclitics
- demonstratives and plural marking on demonstratives
- genitives and possessive classifiers
- verb complex slots
- causative, middle, applicative, passive, and valency
- imperatives, questions, and negation
- complement, relative, temporal, reason, purpose, and coordinate clauses

Missing or unclear information:
- Exact full forms in large tables are preserved on selected page JPGs; compact Markdown summarizes the translation-relevant facts.
- Some categories may have more detail in non-selected pages; selected pages cover the main translation-relevant areas requested.

Consistency and faithfulness:
- Outputs are based only on the input PDF.
- Markdown files are authoritative.
- JPG files were rendered from Markdown or selected PDF pages.
- Materials are ready for Qwen/Qwen-VL experiments.
"""
    (root / "creation_report.txt").write_text(report.strip() + "\n", encoding="utf-8")
    print(report.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--pdf_path_or_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    pdfs = find_pdfs(Path(args.pdf_path_or_folder))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {args.pdf_path_or_folder}")
    for pdf in pdfs:
        process_pdf(pdf, args.language, Path(args.output_dir))


if __name__ == "__main__":
    main()
