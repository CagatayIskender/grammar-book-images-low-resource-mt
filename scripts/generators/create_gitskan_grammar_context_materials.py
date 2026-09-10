from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


import argparse
import os
import re
from pathlib import Path

import fitz

from create_grammar_context_materials import (
    PageInfo,
    clean_pdf_name,
    extract_pages,
    find_pdfs,
    render_markdown_to_jpg,
    render_pdf_page,
    slugify,
    write_both,
)


def contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def first_lines(text: str, n: int = 10) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines[:n])


def pdf_kind(pdf_name: str) -> str:
    if "rigsby" in pdf_name.lower() or "1986" in pdf_name.lower() or "pdf 2" in pdf_name.lower():
        return "rigsby_intro"
    return "ipa"


def score_pages(pages: list[PageInfo], kind: str) -> list[PageInfo]:
    scored = []
    for page in pages:
        score = 0
        reasons = []
        text = page.text
        tests = [
            (5, ["orthographic transcription with interlinear English gloss", "morpheme breakdowns", "interlinear English gloss"], "interlinear glossed text"),
            (5, ["gloss", "IPFV", "PROSP", "AOR", "NEG", "DAT", "ERG", "ABS"], "gloss labels/examples"),
            (4, ["clitic", "enclitic", "affix", "suffix", "prefix", "morpheme"], "morphology/clitics"),
            (4, ["basic constituent ordering", "Verb - Subject - Object", "Subject - Object", "word order"], "constituent order"),
            (4, ["plural", "plurality", "nominal", "verbal morphology", "nominal cases"], "nominal/verbal morphology"),
            (3, ["phonetics", "phonology", "consonants", "vowels", "stress", "syllable"], "phonology useful for forms"),
            (3, ["narrative", "translation", "English", "text"], "text/translation"),
            (3, ["language", "dialect", "Eastern Gitksan", "Western Gitksan"], "language/dialect context"),
            (-5, ["references", "acknowledgements", "bibliography"], "references/front matter"),
        ]
        for weight, terms, label in tests:
            if contains_any(text, terms):
                score += weight
                reasons.append(f"{weight:+d} {label}")
        scored.append(PageInfo(page.number, page.text, score, reasons))
    return scored


def select_pages(pages: list[PageInfo], kind: str) -> list[PageInfo]:
    scored = score_pages(pages, kind)
    if kind == "ipa":
        wanted = [1, 2, 5, 6, 7, 8, 9, 10]
        # Page 11 has the final line of the narrative, but mostly references. Include it only if score remains useful.
        if len(scored) >= 11 and scored[10].score > 0:
            wanted.append(11)
    else:
        wanted = [1, 3, 4, 5, 6, 7]
    selected = [p for p in scored if p.number in wanted and p.score > -2]
    if not selected:
        selected = sorted(scored, key=lambda p: p.score, reverse=True)[: min(8, len(scored))]
    return sorted(selected, key=lambda p: p.number)


def gloss_table_for_kind(kind: str) -> str:
    if kind == "ipa":
        rows = [
            ("IPFV", "imperfective", "appears in narrative gloss yukw=t as IPFV=3.I", "PDF p. 9"),
            ("PROSP", "prospective", "dim is glossed PROSP in narrative examples", "PDF p. 9"),
            ("CN", "connective/clause connector label in glosses", "appears as =hl or gan=hl in narrative glosses", "PDF p. 9"),
            ("CL.CNJ", "clausal conjunction label", "ii=t is glossed CL.CNJ=3.I", "PDF p. 9"),
            ("3.I / 3.II / 3PL / 3SG", "person/number/class labels in glosses", "help identify participants", "PDF p. 9-10"),
            ("CAUS", "causative", "sa-goot-xw is glossed CAUS-heart-PASS", "PDF p. 9"),
            ("PASS", "passive/pass-like label in glosses", "appears in forms such as strong-PASS", "PDF p. 9-10"),
            ("ANTIP", "antipassive", "bax-asxw is glossed run-ANTIP", "PDF p. 9-10"),
            ("NEG", "negation", "nee is glossed NEG", "PDF p. 10"),
            ("EMPH", "emphatic", "ap is glossed EMPH", "PDF p. 10"),
            ("FOC", "focus", "dii is glossed FOC", "PDF p. 10"),
            ("COMP", "complementizer/complement label", "wil is glossed COMP", "PDF p. 10-11"),
        ]
    else:
        rows = [
            ("VSO", "Verb-Subject-Object basic constituent ordering", "suggested as a parallel with Kwakiutlan", "PDF p. 7"),
            ("enclitics", "grammatical relation marking", "precede syntactic host but are suffixed to preceding element", "PDF p. 7"),
            ("focus", "focus constituents move sentence-initially", "used in Tsimshianic languages according to the excerpt", "PDF p. 7"),
            ("plurality", "highly developed grammatical category", "nominal and verbal morphology", "PDF p. 5"),
            ("nominal cases", "structural characterization in quoted Sapir passage", "true nominal cases mentioned for Penutian languages generally", "PDF p. 6"),
        ]
    out = ["| Form/label | Function from PDF | Translation cue | Source page |", "|---|---|---|---|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def summary_content(language: str, pdf_name: str, pages: list[PageInfo], kind: str) -> str:
    display_language = f"{language} (PDF spelling: Gitksan)" if language.lower() != "gitksan" else language
    if kind == "ipa":
        return f"""
# {display_language} Grammar Summary for Translation

## Source
PDF: {pdf_name}
Pages used: selected pages from a 12-page IPA/phonology article, especially pages 1-2 and 5-10.
Extraction notes: embedded PDF text was available. The PDF is mainly phonetics/phonology plus an interlinear narrative text. It is not a full grammar. Full noun case, tense/aspect paradigms, pronoun paradigms, and full syntax chapters are not found in this PDF.

## LANGUAGE_AND_DIALECT_CONTEXT
| Fact | Translation cue | Source page |
|---|---|---|
| Gitksan is identified as an Interior Tsimshianic language spoken in northwestern British Columbia, Canada. | Language identification only; not a direct translation rule. | [PDF p. 1] |
| The dialect presented is Eastern Gitksan, spoken in Kispiox, Glen Vowell, and Hazelton. | Forms in this PDF are Eastern Gitksan. | [PDF p. 1] |
| Eastern and Western dialects differ phonologically by a lexical shift in vowels and stop lenition in Eastern dialects. | Expect dialect-dependent form variation. | [PDF p. 1] |

## WORD_ORDER
| Fact | Translation cue | Source page |
|---|---|---|
| Not found as a general prose rule in this PDF. | Use the interlinear narrative examples rather than an invented rule. | [PDF p. 9-10] |
| Narrative examples show many clitics and particles attached with = in morpheme breakdowns. | Preserve clitic boundaries and use gloss labels to infer roles. | [PDF p. 9-10] |

## PHONOLOGY_AND_ORTHOGRAPHY
| Form/fact | Function | Translation cue | Source page |
|---|---|---|---|
| Gitksan has glottalized plosives/affricates and glottalized sonorants. | phonological inventory | Preserve apostrophe/glottalization in forms. | [PDF p. 2] |
| Practical orthography places glottalization after plosives/affricates and before sonorants. | orthographic convention | Forms such as 'mal or ts'al encode glottalization. | [PDF p. 2], [PDF p. 4] |
| Underlining indicates a uvular consonant in the orthography. | orthographic convention | Uvular contrasts may affect word identity. | [PDF p. 3] |
| Reduced vowel /ə/ appears in affixes and some function words. | vowel behavior | dim is given as [dIm] and glossed PROSP in the narrative. | [PDF p. 5], [PDF p. 9] |
| Vowel length is contrastive. | lexical contrast | Preserve vowel length-related orthographic differences when present. | [PDF p. 5] |
| Stress falls on the rightmost vowel of the root in lexical words. | prosody | Suffixes are stated to be invisible to stress assignment. | [PDF p. 7] |

## MORPHOLOGY_AND_CLITICS
| Form/label | Function | Translation cue | Source page |
|---|---|---|---|
| -t | appears as 3.II or 3.I in narrative glosses depending on context | participant/person-class label in examples | [PDF p. 9-10] |
| =hl | CN in narrative glosses | connective/clause label; preserve as clitic | [PDF p. 9-10] |
| dim | PROSP | prospective meaning in narrative examples | [PDF p. 9] |
| nee | NEG | negation in quoted narrative example | [PDF p. 10] |
| ap | EMPH | emphatic marker in quoted narrative example | [PDF p. 10] |
| dii | FOC | focus label in examples such as nee=dii and hlaa=dii | [PDF p. 10] |
| -diit | 3PL in glossed examples | plural participant label | [PDF p. 9-10] |
| sa- | CAUS in examples | causative label where present in glosses | [PDF p. 9] |
| -xw | PASS in examples | pass/passive-like label where present in glosses | [PDF p. 9-10] |

## VERB
| Fact | Translation cue | Source page |
|---|---|---|
| Full tense/aspect/mood paradigms are not found in this PDF. | Use only labels in the narrative glosses. | [PDF p. 9-10] |
| IPFV appears in yukw=t in the narrative glosses. | imperfective/progressive context in translations such as 'were discussing'. | [PDF p. 9] |
| PROSP appears as dim in the narrative glosses. | prospective/future-like cue in translations. | [PDF p. 9] |
| ANTlP/ANTIP appears in bax-asxw 'run-ANTIP'. | morphology label in examples; do not infer beyond PDF. | [PDF p. 9-10] |

## NEGATION
| Form | Function | Translation cue | Source page |
|---|---|---|---|
| nee | NEG | appears in 'I can't do it' example. | [PDF p. 10] |
| nee=dii | NEG=FOC | negated focus-marked form in narrative example. | [PDF p. 10] |

## QUESTIONS
Not found as a general grammar section in this PDF.

## SUBORDINATION_AND_CLAUSE_CONNECTION
| Form/label | Function | Translation cue | Source page |
|---|---|---|---|
| =hl | CN | connective/clause connector label in many narrative examples. | [PDF p. 9-10] |
| CL.CNJ | clausal conjunction label | ii=t appears as CL.CNJ=3.I in narrative glosses. | [PDF p. 9-10] |
| COMP | complementizer/complement label | wil appears as COMP in narrative examples. | [PDF p. 10-11] |

## GLOSSING_ABBREVIATIONS
{gloss_table_for_kind(kind)}

## TRANSLATION_RELEVANT_EXAMPLES
| Page | Original / gloss / translation |
|---|---|
| [PDF p. 9] | Yukwt laseexwhl bahasxw ganhl hloxs / yukw=t laseexw=hl bax-asxw gan=hl hloxs / IPFV=3.I discuss=CN run-ANTIP PH.CNJ=CN sun / 'The wind and the sun were discussing amongst themselves who was the strongest of them.' |
| [PDF p. 9] | Yukwt laseexwdiit naa dim ky'aa daxgyadit / yukw=t laseexw-diit naa dim ky'aa daxgyat-it / IPFV=3.I discuss-3PL who PROSP most strong-SX / 'They were discussing who was the strongest.' |
| [PDF p. 9] | dis wihl hagwin 'witxwhl lixsgyadit iit hooxhl 'wii gwila / time COMP=CN toward come=CN different-person-SX CL.CNJ=3.I use=CN big blanket / 'Just then a stranger arrived wearing a big blanket.' |
| [PDF p. 10] | nee ap hlgu-xw-s-in-'y / NEG EMPH small-PASS-PASS-CAUS-1SG.II / 'I can't do it'. |
| [PDF p. 10] | Farther narrative examples show nee=dii, hlaa=dii, and COMP wil in context. Exact forms should be checked in the selected JPG. |

## TRANSLATION_CUES_FOR_QWEN
- TRANSLATION_CUE: Use the interlinear glosses directly. This PDF does not provide a full grammar; do not invent case, pronoun, or tense paradigms.
- PARTICLE: dim is glossed PROSP in the narrative and should be treated as a prospective cue when seen in comparable contexts. [PDF p. 9]
- NEGATION: nee is glossed NEG and appears in an English translation with "can't". [PDF p. 10]
- PARTICLE: dii is glossed FOC in nee=dii and hlaa=dii examples. [PDF p. 10]
- SUBORDINATION: wil is glossed COMP in examples; =hl is glossed CN and appears frequently as a clitic boundary. [PDF p. 9-10]
- ORTHOGRAPHY: Preserve apostrophes, underlining/uvular marking if visible, and clitic/morpheme boundaries because they distinguish forms. [PDF p. 2-3], [PDF p. 9-10]

## NOT_FOUND_IN_THIS_PDF
- Full case system: not found in the PDF.
- Full pronoun paradigm: not found in the PDF.
- Full tense/aspect/mood paradigm: not found in the PDF.
- General question formation: not found in the PDF.
- General word order rule: not found in the PDF.
"""

    return f"""
# {display_language} Grammar Summary for Translation

## Source
PDF: {pdf_name}
Pages used: selected pages from a 9-page introductory excerpt, especially pages 1, 3-7.
Extraction notes: embedded PDF text was available, but OCR/text quality is noisy in places. This PDF is mostly sociolinguistic and historical introduction. It contains limited direct grammar information; the most translation-relevant grammar appears mainly on pages 5-7.

## LANGUAGE_AND_SOURCE_CONTEXT
| Fact | Translation cue | Source page |
|---|---|---|
| The monograph is described as a grammatical description of the Gitksan language. | Source identification. | [PDF p. 1] |
| The author discusses whether Gitksan should be treated as language or dialect and notes community norms around calling it a language. | Use "Gitksan language" as the source's framing. | [PDF p. 1-2] |
| The description is based mostly on eastern Gitksan language varieties spoken in Kispiox and Hazelton. | Forms described in this source are mainly eastern Gitksan varieties. | [PDF p. 4] |
| Example sentences in the grammar are said to have been checked by one or more older speakers; many come from observation/texts, most from direct elicitation. | Examples in the grammar are treated as checked source material. | [PDF p. 4] |

## WORD_ORDER
| Fact | Translation cue | Source page |
|---|---|---|
| The excerpt suggests at least a Verb-Subject-Object basic constituent ordering followed by peripheral, oblique constituents in independent clauses. | Expect verb-initial order in independent clauses according to this excerpt. | [PDF p. 7] |
| Grammatical relations of major constituents are indicated not only by ordering, but also by enclitics. | Do not rely on order alone; attend to enclitics. | [PDF p. 7] |
| Tsimshianic languages focus constituents by moving them into sentence-initial position. | Sentence-initial constituent may be focus. | [PDF p. 7] |

## MORPHOLOGY
| Fact | Translation cue | Source page |
|---|---|---|
| Plurality is described as a highly developed grammatical category in Gitksan, Nisgha, and Coast Tsimshian nominal and verbal morphology. | Number/plural marking can be important in both nouns and verbs. | [PDF p. 5] |
| The excerpt mentions plural construction types ranging from simple initial reduplicated forms to opaque/doubly marked reduplications and suppletive sets. | Plurality may not be a single transparent suffix. | [PDF p. 5] |
| The excerpt discusses lexical formations with derivational suffixes as important for comparative study. | Derivational suffixes may be structurally important, but details are not given here. | [PDF p. 7] |

## CASE_AND_GRAMMATICAL_RELATIONS
| Fact | Translation cue | Source page |
|---|---|---|
| A quoted Sapir characterization says Penutian languages possess true nominal cases for the most part; this is not presented as a Gitksan-specific paradigm. | Do not infer a Gitksan case system from this excerpt alone. | [PDF p. 6] |
| For Gitksan/Tsimshianic constructions, grammatical relations are said to be indicated by ordering and enclitics. | Enclitics may help identify roles. | [PDF p. 7] |

## PRONOUNS
Not found in this PDF excerpt.

## VERB
| Fact | Translation cue | Source page |
|---|---|---|
| Basic independent clause order is described as Verb-Subject-Object. | Verb may appear before subject and object. | [PDF p. 7] |
| Detailed verb paradigms are not found in this excerpt. | Do not invent tense/aspect/mood rules. | [PDF p. 7] |

## PARTICLES_AND_CLITICS
| Fact | Translation cue | Source page |
|---|---|---|
| Enclitics precede their syntactic host but are suffixed to the preceding element. | An enclitic may mark the following constituent while appearing on the previous word. | [PDF p. 7] |
| Focus constituents are moved into sentence-initial position. | Initial constituent may indicate focus rather than basic argument position. | [PDF p. 7] |

## NEGATION
Not found in this PDF excerpt.

## QUESTIONS
Not found in this PDF excerpt.

## GLOSSING_ABBREVIATIONS
{gloss_table_for_kind(kind)}

## TRANSLATION_RELEVANT_EXAMPLES
This PDF excerpt does not provide interlinear Gitksan examples with English translations comparable to the IPA article. It does give methodological statements about examples in the larger grammar: example sentences were checked by older speakers; many come from observations/texts, but most from direct elicitation. [PDF p. 4]

## TRANSLATION_CUES_FOR_QWEN
- WORD_ORDER: The excerpt suggests VSO basic constituent ordering in independent clauses, followed by peripheral/oblique constituents. [PDF p. 7]
- PARTICLE: Enclitics may mark grammatical relations and may precede their syntactic host while being suffixed to the preceding element. [PDF p. 7]
- TRANSLATION_CUE: Sentence-initial material may be focus, since focus constituents are described as moving sentence-initially. [PDF p. 7]
- TRANSLATION_CUE: Plurality is described as highly developed in nominal and verbal morphology; plural construction types may include reduplication and suppletion. [PDF p. 5]

## NOT_FOUND_IN_THIS_PDF
- Full case paradigm: not found as a Gitksan-specific paradigm.
- Pronoun paradigm: not found.
- Tense/aspect/mood paradigms: not found.
- Negation and question formation: not found.
- Interlinear examples with translations: not found in this excerpt.
"""


def cheat_content(language: str, pdf_name: str, kind: str) -> str:
    if kind == "ipa":
        return f"""
# {language} Compact Translation Cheat Sheet
Source: {pdf_name}

## USE_LIMIT
This PDF is mostly phonetics/phonology plus a glossed narrative. Do not infer full grammar paradigms.

## HIGH_VALUE_CUES
| Form/label | Function from PDF | Translation cue | Page |
|---|---|---|---|
| dim | PROSP | prospective/future-like cue in narrative | [PDF p. 9] |
| nee | NEG | negation; appears in 'I can't do it' | [PDF p. 10] |
| dii | FOC | focus label in forms like nee=dii | [PDF p. 10] |
| =hl | CN | connective/clitic boundary | [PDF p. 9-10] |
| wil | COMP | complementizer/complement label | [PDF p. 10-11] |
| yukw=t | IPFV=3.I | imperfective/progressive cue | [PDF p. 9] |
| -diit | 3PL | plural participant marker in examples | [PDF p. 9-10] |

## EXAMPLES
| Gloss cue | English translation | Page |
|---|---|---|
| IPFV=3.I discuss=CN ... | 'The wind and the sun were discussing...' | [PDF p. 9] |
| who PROSP most strong-SX | 'who was the strongest' | [PDF p. 9] |
| NEG EMPH small-PASS-PASS-CAUS-1SG.II | 'I can't do it' | [PDF p. 10] |

## FORM_PRESERVATION
Preserve apostrophes/glottalization, clitic =, and morpheme hyphens. The PDF says glottalization is represented in the practical orthography. [PDF p. 2]
"""
    return f"""
# {language} Compact Translation Cheat Sheet
Source: {pdf_name}

## USE_LIMIT
This PDF is an introductory excerpt. It gives limited grammar facts; do not invent missing paradigms.

## HIGH_VALUE_CUES
| Fact | Translation cue | Page |
|---|---|---|
| VSO basic constituent ordering suggested for independent clauses | expect verb before subject/object | [PDF p. 7] |
| Peripheral/oblique constituents follow VSO core | later constituents may be oblique/peripheral | [PDF p. 7] |
| Grammatical relations indicated by ordering and enclitics | use enclitics, not only order | [PDF p. 7] |
| Enclitics precede syntactic host but suffix to preceding element | clitic position may point forward | [PDF p. 7] |
| Focus constituents move sentence-initially | sentence-initial material may be focus | [PDF p. 7] |
| Plurality highly developed in nominal and verbal morphology | plural marking may be important and varied | [PDF p. 5] |

## NOT_FOUND
Pronoun paradigm, negation rules, question formation, full verb paradigms, and interlinear examples are not found in this PDF excerpt.
"""


def tables_content(language: str, pdf_name: str, kind: str) -> str:
    if kind == "ipa":
        return f"""
# {language} Summary Tables for Qwen-VL
Source: {pdf_name}

## SOURCE_SCOPE
| Scope | Use |
|---|---|
| IPA/phonology article | Good for forms, orthography, selected gloss labels, and narrative examples |
| Not full grammar | Do not infer missing paradigms |

## PARTICLES_CLITICS_GLOSS
{gloss_table_for_kind(kind)}

## TRANSLATION_EXAMPLES
| Form/gloss | Translation | Page |
|---|---|---|
| yukw=t laseexw=hl ... | 'were discussing...' | [PDF p. 9] |
| naa dim ky'aa daxgyat-it | 'who was the strongest' | [PDF p. 9] |
| nee ap hlgu-xw-s-in-'y | 'I can't do it' | [PDF p. 10] |
| wil | COMP in narrative examples | [PDF p. 10-11] |

## ORTHOGRAPHY_FORM
| Fact | Cue | Page |
|---|---|---|
| glottalization written with apostrophe position conventions | preserve apostrophes | [PDF p. 2] |
| underlining indicates uvular consonant | preserve visual distinction if possible | [PDF p. 3] |
| vowel length contrastive | keep orthographic distinctions | [PDF p. 5] |
| stress rightmost root vowel in lexical words | suffixes invisible to stress | [PDF p. 7] |
"""
    return f"""
# {language} Summary Tables for Qwen-VL
Source: {pdf_name}

## SOURCE_SCOPE
| Scope | Use |
|---|---|
| Introductory excerpt | Useful for language/dialect context and a few grammar cues |
| Not full grammar | Do not infer missing paradigms |

## WORD_ORDER_AND_RELATIONS
| Fact | Translation cue | Page |
|---|---|---|
| Verb-Subject-Object basic constituent ordering suggested | verb may be clause-initial | [PDF p. 7] |
| Peripheral/oblique constituents follow | post-core constituents may be peripheral | [PDF p. 7] |
| Relations indicated by order and enclitics | inspect clitics | [PDF p. 7] |
| Focus constituents move sentence-initially | initial material may be focus | [PDF p. 7] |

## MORPHOLOGY
| Fact | Translation cue | Page |
|---|---|---|
| Plurality highly developed | number can matter in nominal and verbal morphology | [PDF p. 5] |
| Plural types include reduplication and suppletion | plural may not be transparent suffix | [PDF p. 5] |
| Derivational suffixes mentioned as important for comparative study | details not given | [PDF p. 7] |

## NOT_FOUND
| Missing section | Status |
|---|---|
| Pronouns | Not found in PDF excerpt |
| Negation | Not found in PDF excerpt |
| Questions | Not found in PDF excerpt |
| Full verb paradigms | Not found in PDF excerpt |
"""


def write_manifest(out_path: Path, selected: list[PageInfo], page_images: dict[int, str]) -> str:
    lines = ["# Selected Impactful PDF Pages", ""]
    txt = ["Selected Impactful PDF Pages", ""]
    for page in selected:
        usefulness = "high" if page.score >= 12 else "medium" if page.score >= 7 else "low"
        reason = "; ".join(page.reasons) if page.reasons else "selected by fixed page policy for this PDF"
        block = [
            f"## PDF page {page.number}",
            f"- JPG filename: {page_images[page.number]}",
            f"- Selection score: {page.score}",
            f"- Expected usefulness: {usefulness}",
            f"- Reason selected: {reason}",
            f"- Grammar information: {first_lines(page.text, 6)}",
            "- Readability/OCR concerns: embedded text extracted; some scanned/OCR text is noisy, especially in the Rigsby excerpt.",
            "",
        ]
        lines.extend(block)
        txt.extend([line.replace("#", "").strip() for line in block])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.with_suffix(".txt").write_text("\n".join(txt), encoding="utf-8")
    return ", ".join(str(p.number) for p in selected)


def process_pdf(pdf_path: Path, language: str, output_dir: Path) -> None:
    pdf_name = clean_pdf_name(pdf_path)
    kind = pdf_kind(pdf_name)
    root = output_dir.resolve() / "materials/curated_v1" / ("gitksan" if language.lower() in ("gitskan", "gitksan") else language.lower()) / (("pdf1_brown" if "PDF 1" in pdf_path.name else "pdf2_rigsby") if language.lower() in ("gitskan", "gitksan") else "grammar") / "original"
    summary_dir = root / "summary_text"
    tables_dir = root / "summary_tables"
    cheat_dir = root / "cheatsheet"
    pages_dir = root / "page_notes"
    selected_dir = root / "selected_pages"

    for d in [summary_dir, tables_dir, cheat_dir, pages_dir, selected_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for old in selected_dir.glob("page_*.jpg"):
        old.unlink()

    pages = score_pages(extract_pages(pdf_path), kind)
    selected = select_pages(pages, kind)
    doc = fitz.open(pdf_path)

    page_images = {}
    render_notes = []
    for page in selected:
        filename = f"page_{page.number:03d}.jpg"
        size = render_pdf_page(pdf_path, page.number, selected_dir / filename, zoom=2.0)
        page_images[page.number] = filename
        render_notes.append(f"PDF page {page.number} -> {filename}, size={size[0]}x{size[1]}, score={page.score}")

    summary = summary_content(language, pdf_name, pages, kind)
    write_both(summary_dir / "summary_model_readable.md", summary_dir / "summary_model_readable.txt", summary)

    cheat = cheat_content(language, pdf_name, kind)
    cheat_md = cheat_dir / "compact_cheatsheet.md"
    write_both(cheat_md, cheat_dir / "compact_cheatsheet.txt", cheat)
    cheat_images = render_markdown_to_jpg(cheat_md, cheat_dir / "compact_cheatsheet.jpg")

    tables = tables_content(language, pdf_name, kind)
    tables_md = tables_dir / "summary_tables.md"
    write_both(tables_md, tables_dir / "summary_tables.txt", tables)
    table_images = render_markdown_to_jpg(tables_md, tables_dir / "summary_tables.jpg")

    selected_pages = write_manifest(selected_dir / "selected_pages_manifest.md", selected, page_images)

    notes = [
        "# Optional Page Notes",
        "",
        f"PDF file: {pdf_path}",
        f"Total pages: {doc.page_count}",
        "Embedded text extraction was used.",
        "OCR was not run.",
        "Only selected impactful pages were rendered to JPG.",
        "The Rigsby excerpt has noisy extracted text because it appears scanned/OCR-like; exact forms should be checked in selected JPGs.",
        "The IPA article is mostly phonetics/phonology; grammar facts are mainly from its interlinear narrative.",
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

    conv = [
        "Page conversion notes",
        f"PDF file: {pdf_path}",
        f"Total PDF pages: {doc.page_count}",
        f"Selected pages rendered: {len(selected)}",
        "All pages were not rendered because not every page is translation-relevant.",
        "Rendering: PyMuPDF zoom=2.0, JPEG quality=82.",
        "Failures: none.",
    ]
    (pages_dir / "page_conversion_notes.txt").write_text("\n".join(conv) + "\n", encoding="utf-8")

    major_areas = (
        "- orthography/phonology useful for preserving forms\n"
        "- interlinear gloss labels from narrative text\n"
        "- negation/focus/prospective/connective labels\n"
        "- selected translated narrative examples"
        if kind == "ipa"
        else "- language/dialect and source context\n"
        "- VSO constituent order cue\n"
        "- enclitic grammatical relation cue\n"
        "- focus-fronting cue\n"
        "- plurality in nominal and verbal morphology"
    )

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
{major_areas}

Missing or unclear information:
- This PDF is not a complete grammar context source.
- Missing categories are marked as not found in the generated summaries.
- Some extracted text may be noisy, especially in scanned/OCR-like pages.

Consistency and faithfulness:
- Outputs are based only on the input PDF.
- Markdown files are authoritative.
- JPG files were rendered from Markdown or selected PDF pages.
- Materials are ready for Qwen/Qwen-VL experiments with the limitations recorded above.
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
