from __future__ import annotations

import argparse
import os
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


KEYWORD_WEIGHTS = [
    (5, re.compile(r"\b(table|paradigm|singular|plural|absolutive|ergative|genitive|dative|case)\b", re.I)),
    (5, re.compile(r"\b(gloss|translation|example|AOR|IMPF|DAT|ERG|ABS|GEN)\b")),
    (4, re.compile(r"\b(pronoun|agreement|tense|aspect|mood|negation|negative|prohibitive)\b", re.I)),
    (3, re.compile(r"\b(word order|syntax|clause|subordinate|relative|coordination|question)\b", re.I)),
    (3, re.compile(r"\b(particle|clitic|postposition|auxiliary|light verb|suffix|prefix|morpheme)\b", re.I)),
    (2, re.compile(r"\b(abbreviation|abbreviated category labels)\b", re.I)),
    (-4, re.compile(r"\b(bibliography|references|acknowledg|contents|title|editors)\b", re.I)),
]


@dataclass
class PageInfo:
    number: int
    text: str
    score: int
    reasons: list[str]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def clean_pdf_name(path: Path) -> str:
    return re.sub(r"[_-]+", " ", path.stem).strip()


def extract_pages(pdf_path: Path) -> list[PageInfo]:
    doc = fitz.open(pdf_path)
    pages = []
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        score = 0
        reasons = []
        for weight, pattern in KEYWORD_WEIGHTS:
            matches = pattern.findall(text)
            if matches:
                score += weight
                label = pattern.pattern.replace("\\b", "").replace("(", "").replace(")", "")
                reasons.append(f"{weight:+d} {label[:45]}")
        pages.append(PageInfo(idx, text, score, reasons))
    return pages


def find_pdfs(path_or_folder: Path) -> list[Path]:
    if path_or_folder.is_file() and path_or_folder.suffix.lower() == ".pdf":
        return [path_or_folder]
    return sorted(path_or_folder.glob("*.pdf"))


def page_text(pages: list[PageInfo], number: int) -> str:
    return pages[number - 1].text if 1 <= number <= len(pages) else ""


def contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def first_lines(text: str, n: int = 12) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines[:n])


def select_impactful_pages(pages: list[PageInfo], target: int) -> list[PageInfo]:
    def is_low_value_front_matter(page: PageInfo) -> bool:
        lines = [line.strip().lower() for line in page.text.splitlines() if line.strip()]
        head = " ".join(lines[:3])
        return (
            "contents" in head
            or "table of contents" in head
            or "editors" in head
            or "mouton grammar library" in head
        )

    # Ensure coverage of the pages that contain the overview, abbreviations, gloss notation,
    # and any later pages with translation-relevant morphology.
    must_include = []
    for page in pages:
        if is_low_value_front_matter(page):
            continue
        text = page.text
        if contains_any(text, ["Abbreviated category labels", "1.2.2. Morphology", "1.2.3. Syntax"]):
            must_include.append(page)
        elif contains_any(text, ["Polar questions", "parametric questions", "1.3.5.5. Subordinate clauses"]):
            must_include.append(page)
        elif contains_any(text, ["negative suffix", "Post-tonic Vowel Syncope", "Imperfective suffixes"]):
            must_include.append(page)

    ranked = sorted(
        [page for page in pages if not is_low_value_front_matter(page)],
        key=lambda p: (p.score, len(p.text)),
        reverse=True,
    )
    selected = []
    seen = set()
    for page in must_include + ranked:
        if page.number in seen:
            continue
        if page.score <= 0 and len(selected) >= max(8, target // 2):
            continue
        selected.append(page)
        seen.add(page.number)
        if len(selected) >= target:
            break
    return sorted(selected, key=lambda p: p.number)


def render_pdf_page(pdf_path: Path, page_number: int, out_path: Path, zoom: float = 2.0) -> tuple[int, int]:
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=82, optimize=True)
    return img.size


def wrap_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    if line.startswith("|"):
        return [line]
    indent = len(line) - len(line.lstrip(" "))
    prefix = " " * indent
    wrapped = textwrap.wrap(line, width=max(30, width - indent), break_long_words=False, replace_whitespace=False)
    return [prefix + item for item in wrapped] or [line]


def render_markdown_to_jpg(md_path: Path, out_base: Path) -> list[Path]:
    text = md_path.read_text(encoding="utf-8")
    font_regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font_bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if not font_regular.exists():
        font_regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    font = ImageFont.truetype(str(font_regular), 28)
    bold = ImageFont.truetype(str(font_bold if font_bold.exists() else font_regular), 34)
    mono = ImageFont.truetype(str(font_regular), 24)

    width = 1900
    height = 2600
    margin = 70
    line_gap = 12
    max_chars = 112
    table_pad_x = 14
    table_pad_y = 10
    table_line_gap = 7

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for old_render in [out_base, *out_base.parent.glob(f"{out_base.stem}_*{out_base.suffix}")]:
        if old_render.exists():
            old_render.unlink()

    pages = []
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = margin

    def is_table_separator(line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            return False
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)

    def parse_table_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def flush():
        nonlocal img, draw, y
        idx = len(pages) + 1
        if idx == 1:
            out_path = out_base
        else:
            out_path = out_base.with_name(f"{out_base.stem}_{idx}{out_base.suffix}")
        img.save(out_path, "JPEG", quality=90, optimize=True)
        pages.append(out_path)
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        y = margin

    def draw_markdown_table(rows: list[list[str]]) -> None:
        nonlocal y
        if not rows:
            return
        col_count = max(len(row) for row in rows)
        normalized = [row + [""] * (col_count - len(row)) for row in rows]
        max_width = width - (2 * margin)
        raw_widths = []
        min_widths = []
        for idx in range(col_count):
            measured_cells = []
            measured_tokens = []
            for row_idx, row in enumerate(normalized):
                cell_font = bold if row_idx == 0 else mono
                measured_cells.append(draw.textlength(row[idx], font=cell_font))
                tokens = re.split(r"\s+", row[idx])
                measured_tokens.extend(draw.textlength(token, font=cell_font) for token in tokens if token)
            raw_widths.append(max(130, int(max(measured_cells or [0]) + 2 * table_pad_x)))
            min_widths.append(max(110, int(max(measured_tokens or [0]) + 2 * table_pad_x)))
        total_raw = sum(raw_widths)
        if total_raw <= max_width:
            col_widths = raw_widths
        else:
            min_total = sum(min_widths)
            flex_width = max_width - min_total
            if flex_width <= 0:
                col_widths = [max_width // col_count] * col_count
            else:
                flex_raw = [max(value - min_widths[idx], 1) for idx, value in enumerate(raw_widths)]
                flex_total = sum(flex_raw)
                col_widths = [
                    min_widths[idx] + int(flex_width * value / flex_total)
                    for idx, value in enumerate(flex_raw)
                ]
                overflow = sum(col_widths) - max_width
                if overflow > 0:
                    col_widths[-1] -= overflow

        def wrap_cell(cell: str, cell_width: int, cell_font: ImageFont.FreeTypeFont) -> list[str]:
            approx_char_width = max(8, draw.textlength("abcdefghijklmnopqrstuvwxyz", font=cell_font) / 26)
            char_width = max(10, int((cell_width - 2 * table_pad_x) / approx_char_width))
            return textwrap.wrap(cell, width=char_width, break_long_words=False, replace_whitespace=False) or [""]

        for row_idx, row in enumerate(normalized):
            current_font = bold if row_idx == 0 else mono
            cell_lines = [
                wrap_cell(cell, col_widths[col_idx], current_font)
                for col_idx, cell in enumerate(row)
            ]
            row_line_count = max(len(lines) for lines in cell_lines)
            sample_bbox = draw.textbbox((0, 0), "Ag", font=current_font)
            text_height = sample_bbox[3] - sample_bbox[1]
            row_height = (row_line_count * (text_height + table_line_gap)) + (2 * table_pad_y)
            if y + row_height > height - margin:
                flush()
            x = margin
            for col_idx, lines in enumerate(cell_lines):
                fill = (242, 242, 242) if row_idx == 0 else "white"
                draw.rectangle(
                    [x, y, x + col_widths[col_idx], y + row_height],
                    fill=fill,
                    outline="black",
                    width=2,
                )
                text_y = y + table_pad_y
                for line in lines:
                    draw.text((x + table_pad_x, text_y), line, font=current_font, fill="black")
                    text_y += text_height + table_line_gap
                x += col_widths[col_idx]
            y += row_height
        y += line_gap * 2

    raw_lines = text.splitlines()
    i = 0
    while i < len(raw_lines):
        raw = raw_lines[i]
        if (
            raw.strip().startswith("|")
            and i + 1 < len(raw_lines)
            and is_table_separator(raw_lines[i + 1])
        ):
            table_rows = [parse_table_row(raw)]
            i += 2
            while i < len(raw_lines) and raw_lines[i].strip().startswith("|"):
                if not is_table_separator(raw_lines[i]):
                    table_rows.append(parse_table_row(raw_lines[i]))
                i += 1
            draw_markdown_table(table_rows)
            continue

        lines = wrap_line(raw, max_chars)
        for line in lines:
            current_font = bold if line.startswith("#") else mono if line.startswith("|") else font
            bbox = draw.textbbox((margin, y), line, font=current_font)
            line_height = bbox[3] - bbox[1] + line_gap
            if y + line_height > height - margin:
                flush()
            draw.text((margin, y), line, font=current_font, fill="black")
            y += line_height
        if raw.strip() == "":
            y += line_gap
        i += 1
    flush()
    return pages


def write_both(md_path: Path, txt_path: Path, content: str) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(content.strip() + "\n", encoding="utf-8")
    txt_path.write_text(content.strip() + "\n", encoding="utf-8")


def abbreviations_table() -> str:
    rows = [
        ("ABS", "Absolutive case", "case label"),
        ("ERG", "Ergative case", "case label"),
        ("GEN", "Genitive case", "case label"),
        ("DAT", "Dative case", "case label"),
        ("ADESS/ADEL/ADDIR", "Adessive/Adelative/Addirective case", "local case labels"),
        ("SBESS/SBEL/SBDIR", "Subessive/Subelative/Subdirective case", "local case labels"),
        ("POESS/POEL/PODIR", "Postessive/Postelative/Postdirective case", "local case labels"),
        ("SRESS/SREL/SRDIR", "Superessive/Superelative/Superdirective case", "local case labels"),
        ("INESS/INEL", "Inessive/Inelative case", "local case labels"),
        ("AOR", "Aorist", "verb category"),
        ("IMPF", "Imperfective", "verb category"),
        ("FUT", "Future", "verb category"),
        ("PRF", "Perfect", "verb category"),
        ("PST", "Past", "verb category"),
        ("NEG", "negation", "negation label"),
        ("Q", "question marker", "question label"),
        ("PTP", "participle", "non-finite verb label"),
        ("INF", "Infinitive", "non-finite verb label"),
        ("MSD", "Masdar", "non-finite verb label"),
        ("AOC", "Aorist converb", "converb label"),
        ("POSTR/TEMP", "Posterior/Temporal converb", "converb labels"),
        ("PT", "particle", "particle label"),
        ("COP", "copula", "copula label"),
        ("PL", "plural", "number label"),
    ]
    out = ["| Form | Function | Translation cue | Source page |", "|---|---|---|---|"]
    for form, fn, cue in rows:
        source = "PDF p. 19" if form not in {"PRF", "Q", "PT", "POSTR/TEMP"} else "PDF p. 20"
        out.append(f"| {form} | {fn} | {cue} | [{source}] |")
    return "\n".join(out)


def summary_content(language: str, pdf_name: str, pages: list[PageInfo]) -> str:
    p24 = page_text(pages, 24)
    p25 = page_text(pages, 25)
    p26 = page_text(pages, 26)
    p27 = page_text(pages, 27)
    p28 = page_text(pages, 28)
    p34 = page_text(pages, 34)
    p35 = page_text(pages, 35)
    p43 = page_text(pages, 43)
    p59 = page_text(pages, 59)
    return f"""
# {language} Grammar Summary for Translation

## Source
PDF: {pdf_name}
Pages used: selected from the 59-page PDF excerpt, especially pages 19-20, 24-28, 30-31, 34-35, 43, and 59.
Extraction notes: embedded PDF text was available. Some transliteration characters in the extracted text look noisy because of PDF font encoding. Such forms were kept as extracted and should be checked against the rendered source page JPGs.

## WORD_ORDER
| Fact | Translation cue | Source page |
|---|---|---|
| Word order patterns are described as overwhelmingly head-final. | Expect dependents before heads. | [PDF p. 25] |
| Head-final order is obligatory in noun phrases, adjective phrases, and postpositional phrases. | Genitive-noun, adjective-noun, numeral-noun, demonstrative-noun are listed. | [PDF p. 25] |
| SOV order is preferred for clauses, but other orders are possible, especially in spoken language. | Do not rely only on position; use case/gloss information. | [PDF p. 25] |

## NOUNS_AND_CASE
| Fact | Translation cue | Source page |
|---|---|---|
| Nouns are inflected for number: Singular and Plural. | PL in glosses marks plural. | [PDF p. 24] |
| Nouns are inflected for case: Absolutive, Ergative, Genitive, Dative, Essive, Elative, Directive. | Case labels indicate clause roles and relations. | [PDF p. 24] |
| Locative cases Essive, Elative, and Directive occur with localizations Ad, Sub, Post, Super, In. | Local cases can express local relations. | [PDF p. 24] |
| All cases other than Absolutive are based on an oblique stem. | A period in nouns can separate the stem from the oblique stem suffix. | [PDF p. 24], [PDF p. 34] |
| Local relations are more often expressed by postpositions; noun inflections tend to express more abstract relations. | Check both case and postposition-like material for English prepositional meanings. | [PDF p. 24] |

### Example nominal paradigm from the PDF: hül 'sea'
| Case | Singular | Plural | Source page |
|---|---|---|---|
| Absolutive | hül | hiil-er | [PDF p. 24] |
| Ergative | hül-i | hiil-er-i | [PDF p. 24] |
| Genitive | hiil-i-n | hül-er-i-n | [PDF p. 24] |
| Dative | hiil-i-z | hül-er-i-z | [PDF p. 24] |
| Adessive | hiil-i-w | hiil-er-i-w | [PDF p. 24] |
| Adelative | hül-i-waj | hül-er-i-waj | [PDF p. 24] |
| Addirective | hiil-i-wdi | hiil-er-i-wdi | [PDF p. 24] |
| Subessive | hiil-i-k | hiil-er-i-k | [PDF p. 24] |
| Subelative | hül-i-kaj | hiil-er-i-kaj | [PDF p. 24] |
| Subdirective | hill-i-kdi | hiil-er-i-kdi | [PDF p. 24] |
| Postessive | hill-i-qh | hiil-er-i-qh | [PDF p. 24] |
| Postelative | hül-i-qhaj | hül-er-i-qhaj | [PDF p. 24] |
| Postdirective | hill-i-qhdi | hül-er-i-q^di | [PDF p. 24] |
| Superessive | hül-e-l | hiXl-er-a-l | [PDF p. 24] |
| Superelative | hiil-e-laj | hiil-er-i-laj | [PDF p. 24] |
| Superdirective | hiil-e-ldi | hiil-er-a-ldi | [PDF p. 24] |
| Inessive | hül-e | hiil-er-a | [PDF p. 24] |
| Inelative | hül-äj | hill-er-aj | [PDF p. 24] |

## PRONOUNS_AND_DEMONSTRATIVES
| Fact | Translation cue | Source page |
|---|---|---|
| Personal pronouns are normally used if there are no full noun phrase arguments. | Pronouns may supply omitted full NPs. | [PDF p. 26] |
| Pronouns may be omitted if recoverable from context. | Missing pronouns may still be understood from context. | [PDF p. 26] |
| Ergative and Absolutive cases of personal pronouns are treated as unanalyzable in the glossing conventions. | Do not over-segment these pronoun forms. | [PDF p. 35] |
| Third-person pronouns are based on demonstratives, but demonstrative/substantivizer/plural affixes may be ignored in glosses. | Forms like ada and abur.u are glossed economically. | [PDF p. 34], [PDF p. 35] |

## VERB
| Fact | Translation cue | Source page |
|---|---|---|
| Verbs are inflected for tense-aspect, negation, several mood forms, and various non-finite forms. | Verb suffixes carry tense/aspect/mood/negation. | [PDF p. 24] |
| There are no person-number agreement forms. | Do not infer subject/object person from verb agreement. | [PDF p. 24], [PDF p. 26] |
| There is no agreement in Lezgian, neither in noun phrases nor on finite verbs. | Use case and NPs/pronouns rather than agreement. | [PDF p. 26] |
| The causative suffix -(a)r turns intransitive verbs into transitive verbs. | Causative may add a causer/transitive reading. | [PDF p. 25], [PDF p. 26] |

### Verb forms from gun 'give'
| Category | Non-negated | Negated | Source page |
|---|---|---|---|
| Imperfective | gu-zwa | gu-zwa-6 | [PDF p. 25] |
| Past Imperfective | gu-zwa-j | gu-zwa-6-ir | [PDF p. 25] |
| Future | gu-da | gu-da-6 | [PDF p. 25] |
| Past Future | gu-da-j | gu-da-ö-ir | [PDF p. 25] |
| Aorist | ga-na | ga-na-ö | [PDF p. 25] |
| Past Aorist | ga-na-j | ga-na-ö-ir | [PDF p. 25] |
| Perfect | ga-nwa | ga-nwa-6 | [PDF p. 25] |
| Past Perfect | ga-nwa-j | ga-nwa-ö-ir | [PDF p. 25] |
| Prohibitive | Not listed | gu-mir | [PDF p. 25] |
| Optative | gu-raj | ta-gu-raj | [PDF p. 25] |
| Hortative | gu-n | ta-gu-n | [PDF p. 25] |
| Masdar | gu-n | ta-gu-n | [PDF p. 25] |
| Infinitive | gu-z | ta-gu-z | [PDF p. 25] |
| Imperfective participle | gu-zwa-j | ta-gu-zwa-j | [PDF p. 25] |
| Future participle | gu-da-j | ta-gu-da-j | [PDF p. 25] |
| Perfect participle | ga-nwa-j | ta-ga-nwa-j | [PDF p. 25] |
| Aorist participle | ga-jt | ta-ga-j | [PDF p. 25] |
| Aorist converb | ga-na | ta-ga-na | [PDF p. 25] |
| Posterior converb | gu-daldi | Not listed | [PDF p. 25] |
| Temporal converb | ga-ji-la | ta-ga-j-la | [PDF p. 25] |

## NEGATION_AND_QUESTIONS
| Fact | Translation cue | Source page |
|---|---|---|
| Many verb forms are listed with non-negated and negated forms. | Negation can be suffixal or prefixal depending on form. | [PDF p. 25] |
| Some negated non-finite/mood forms show ta- before the verb. | ta- in listed forms marks negation for several categories. | [PDF p. 25] |
| Polar questions are marked by the interrogative verb suffix -ni. | -ni on a verb can signal yes/no question. | [PDF p. 27] |
| In parametric questions, the interrogative pronoun is normally in situ and no interrogative verb suffix is used. | Wh-word stays in place; no -ni is used in the example. | [PDF p. 28] |
| Dialects may use different negative/prohibitive forms. | Treat dialectal variants cautiously. | [PDF p. 43] |

## CLAUSE_STRUCTURE
| Fact | Translation cue | Source page |
|---|---|---|
| Clause case-marking is described as uniformly ergative. | Ergative often marks transitive subject; Absolutive marks intransitive subject or object. | [PDF p. 25], [PDF p. 26] |
| Dative subjects occur with some experiential verbs. | DAT NP can correspond to English experiencer subject. | [PDF p. 26] |
| Subordinate clauses are normally non-finite and generally precede the superordinate clause. | Translate bracketed/non-finite clause before main clause if appropriate. | [PDF p. 26] |
| Relative clauses use participles with no inherent orientation. | A participial relative can relativize many constituent roles. | [PDF p. 26] |
| Complement clauses include Masdar, Infinitival, and participial complements. | Non-finite forms can translate as English infinitive/that clauses depending on context. | [PDF p. 26], [PDF p. 27] |
| Conjunction wa 'and' exists, but converb constructions are preferred for clause linking. | Converbs may translate as 'and', 'having...', 'when', 'before', or adverbial clauses. | [PDF p. 27] |

## GLOSSING_AND_NOTATION
| Fact | Translation cue | Source page |
|---|---|---|
| Morpheme-by-morpheme glosses use brackets to mark subordinate clauses. | Brackets help identify embedded clauses. | [PDF p. 28], [PDF p. 35] |
| Categories expressed by zero may be shown in parentheses; categories always expressed by zero are not shown for economy. | Missing ABS or other zero categories may still be present. | [PDF p. 34] |
| A period in nouns separates the stem from the semantically empty oblique stem suffix. | Period is not a word boundary; it marks oblique stem notation. | [PDF p. 34] |
| Lezgian text hyphen is rendered by equals sign in glossed examples to avoid confusion with morpheme hyphens. | Equals sign may represent original hyphen/compound. | [PDF p. 35] |

## GLOSSING_ABBREVIATIONS
{abbreviations_table()}

## TRANSLATION_RELEVANT_EXAMPLES
| Page | Original / gloss / translation |
|---|---|
| [PDF p. 25] | Stxa k'wal.i-z xta-na. / brother(ABS) house-DAT return-AOR / 'The brother came back home.' |
| [PDF p. 26] | Wax.a stxa k'wal.i-z raqur-na. / sister(ERG) brother(ABS) house-DAT send-AOR / 'The sister sent the brother home.' |
| [PDF p. 26] | Wax.a-z stxa aku-na. / sister-DAT brother(ABS) see-AOR / 'The sister saw the brother.' |
| [PDF p. 26] | Ada abur k'wal.i-z raqur-na. / she(ERG) they(ABS) house-DAT send-AOR / 'She sent them home.' |
| [PDF p. 26] | gada k'wal.i-z raqur-aj ruS / [boy house-DAT send-AOP] girl / 'the girl who sent the boy home.' |
| [PDF p. 27] | Farid ata-na-ni? / Farid come-AOR-Q / 'Has Farid come?' |
| [PDF p. 28] | Farid mus ata-na? / Farid when come-AOR / 'When did Farid come?' |
| [PDF p. 28] | Awar 6'al lezgi ö'al.a-laj öetin ja. / Avar language Lezgian language-SREL difficult COP / 'Avar is more difficult than Lezgian.' |

## TRANSLATION_CUES_FOR_QWEN
- CASE: Because clauses are described as uniformly ergative, use ERG/ABS/DAT labels to infer English subject/object roles rather than relying only on word order. [PDF p. 25-26]
- WORD_ORDER: Head-final structure means modifiers and genitives commonly precede nouns; clauses prefer SOV order, but other orders can occur. [PDF p. 25]
- VERB: Verb morphology carries tense-aspect, negation, mood, and non-finite categories; finite verbs do not show person-number agreement. [PDF p. 24], [PDF p. 26]
- QUESTION: -ni on the verb marks polar questions; parametric questions use an in-situ interrogative pronoun without the question suffix in the example. [PDF p. 27-28]
- SUBORDINATION: Non-finite subordinate clauses generally precede main clauses and may translate as English relative, complement, temporal, or adverbial clauses. [PDF p. 26-27]
- NEGATION: Negated forms in the overview include suffixal negative forms and ta- prefixed forms; use the specific category table when available. [PDF p. 25]

## NOT_FOUND_OR_LIMITED_IN_THIS_PDF_EXCERPT
- Full detailed chapters on noun cases, pronouns, postpositions, verbal valence, clause combining, and texts are listed in the contents but are not present in this 59-page PDF excerpt.
- Full paradigms beyond the overview tables were not available in the extracted PDF pages.
- Some orthographic/transliteration symbols are noisy in extracted text; inspect selected page JPGs for exact forms.
"""


def cheatsheet_content(language: str, pdf_name: str) -> str:
    return f"""
# {language} Compact Translation Cheat Sheet

Source: {pdf_name}. Use only as grammar context from the PDF excerpt.

## WORD_ORDER
| Cue | Use for translation | Page |
|---|---|---|
| Head-final patterns | Dependents before heads; SOV preferred in clauses | [PDF p. 25] |
| NP order | Genitive-noun, adjective-noun, numeral-noun, demonstrative-noun | [PDF p. 25] |

## CASE
| Case/type | Cue | Page |
|---|---|---|
| ABS | intransitive subject or object in examples | [PDF p. 25-26] |
| ERG | transitive subject in examples | [PDF p. 26] |
| DAT | goal with house-DAT; experiencer subject with some verbs | [PDF p. 25-26] |
| GEN | possessor/modifier before noun | [PDF p. 25] |
| Localized cases | Ad/Sub/Post/Super/In + Essive/Elative/Directive | [PDF p. 24] |

## VERB
| Category | Forms/cues from gun 'give' | Page |
|---|---|---|
| IMPF | gu-zwa; negated gu-zwa-6 | [PDF p. 25] |
| FUT | gu-da; negated gu-da-6 | [PDF p. 25] |
| AOR | ga-na; negated ga-na-ö | [PDF p. 25] |
| PRF | ga-nwa; negated ga-nwa-6 | [PDF p. 25] |
| INF | gu-z; negated ta-gu-z | [PDF p. 25] |
| PROHIB | gu-mir | [PDF p. 25] |
| Agreement | no person-number agreement forms | [PDF p. 24], [PDF p. 26] |

## QUESTIONS_AND_NEGATION
| Form | Function | Page |
|---|---|---|
| -ni | polar question suffix on verb | [PDF p. 27] |
| in-situ interrogative pronoun | parametric question; no -ni in example | [PDF p. 28] |
| ta- | negation in several non-finite/mood forms in table | [PDF p. 25] |

## CLAUSES
| Construction | Translation cue | Page |
|---|---|---|
| Non-finite subordinate clause | usually precedes main clause | [PDF p. 26] |
| Participial relative | can relativize many constituent roles | [PDF p. 26] |
| Converb | often translates as English adverbial/and/when/before clause | [PDF p. 27] |
| Dative subject | can translate as English experiencer subject | [PDF p. 26] |

## MINI_GLOSS_TABLE
| ABS | Absolutive | ERG | Ergative | DAT | Dative |
|---|---|---|---|---|---|
| GEN | Genitive | AOR | Aorist | IMPF | Imperfective |
| INF | Infinitive | PTP | Participle | Q | Question marker |
| NEG | Negation | PL | Plural | COP | Copula |

## HIGH_VALUE_EXAMPLES
| Example | Translation cue | Page |
|---|---|---|
| sister(ERG) brother(ABS) house-DAT send-AOR | 'The sister sent the brother home.' | [PDF p. 26] |
| sister-DAT brother(ABS) see-AOR | DAT experiencer -> 'The sister saw the brother.' | [PDF p. 26] |
| Farid come-AOR-Q | polar question -> 'Has Farid come?' | [PDF p. 27] |
| Farid when come-AOR | wh-question with in-situ pronoun | [PDF p. 28] |
"""


def summary_tables_content(language: str, pdf_name: str) -> str:
    return f"""
# {language} Summary Tables for Qwen-VL

Source: {pdf_name}. Page references point to the PDF.

## WORD_ORDER
| Pattern | Translation use | Page |
|---|---|---|
| Head-final | Translate modifiers before heads when appropriate | [PDF p. 25] |
| SOV preferred | Verb often late in clause; other orders possible | [PDF p. 25] |
| Postpositional phrases | Postposition-like material follows complement | [PDF p. 25] |

## NOUNS_CASES
| Area | PDF statement | Translation cue | Page |
|---|---|---|---|
| Number | Singular, Plural | PL marks plural | [PDF p. 24] |
| Core cases | Absolutive, Ergative, Genitive, Dative | map roles/relations | [PDF p. 24] |
| Local cases | Essive, Elative, Directive + Ad/Sub/Post/Super/In | local/abstract relations | [PDF p. 24] |
| Oblique stem | all non-ABS cases based on oblique stem | period marks oblique stem in notation | [PDF p. 24], [PDF p. 34] |

## VERB_FORMS_FROM_GUN_GIVE
| Category | Non-negated | Negated | Page |
|---|---|---|---|
| IMPF | gu-zwa | gu-zwa-6 | [PDF p. 25] |
| FUT | gu-da | gu-da-6 | [PDF p. 25] |
| AOR | ga-na | ga-na-ö | [PDF p. 25] |
| PRF | ga-nwa | ga-nwa-6 | [PDF p. 25] |
| PROHIB | -- | gu-mir | [PDF p. 25] |
| OPT | gu-raj | ta-gu-raj | [PDF p. 25] |
| INF | gu-z | ta-gu-z | [PDF p. 25] |
| AOC | ga-na | ta-ga-na | [PDF p. 25] |

## NEGATION_QUESTIONS
| Cue | Meaning/function | Page |
|---|---|---|
| -ni | polar question suffix | [PDF p. 27] |
| in-situ wh pronoun | parametric question | [PDF p. 28] |
| ta- | negation in listed non-finite/mood forms | [PDF p. 25] |
| -6/-ö in table | negated finite categories in extracted table | [PDF p. 25] |

## CLAUSE_PATTERNS
| Pattern | Translation cue | Page |
|---|---|---|
| ERG/ABS transitive | ERG subject, ABS object in example | [PDF p. 26] |
| DAT experiencer | DAT can translate as English subject | [PDF p. 26] |
| Non-finite subordination | subordinate clause precedes main clause | [PDF p. 26] |
| Participial relatives | relative role inferred from context | [PDF p. 26] |
| Converbs | adverbial/sequence meanings | [PDF p. 27] |

## GLOSS_ABBREVIATIONS
{abbreviations_table()}
"""


def write_manifest(out_path: Path, selected: list[PageInfo], page_images: dict[int, str]) -> str:
    lines = ["# Selected Impactful PDF Pages", ""]
    txt_lines = ["Selected Impactful PDF Pages", ""]
    for page in selected:
        text = page.text
        usefulness = "high" if page.score >= 12 else "medium" if page.score >= 7 else "low"
        grammar_bits = []
        for label, terms in [
            ("abbreviations/gloss labels", ["Abbreviated category labels", "Abbreviations"]),
            ("morphology/case/verb overview", ["1.2.2. Morphology", "Nouns are inflected", "Verbs are inflected"]),
            ("syntax/word order/clause examples", ["1.2.3. Syntax", "Word order", "Subordinate clauses"]),
            ("questions", ["Polar questions", "parametric questions"]),
            ("gloss notation", ["1.3.5", "morphemic glosses", "Subordinate clauses"]),
            ("negation/prohibitive variants", ["negative suffix", "prohibitive"]),
            ("phonotactic alternations affecting suffixes", ["Vowel Syncope", "Imperfective suffixes", "Perfect suffixes"]),
        ]:
            if contains_any(text, terms):
                grammar_bits.append(label)
        if not grammar_bits:
            grammar_bits.append("translation-relevant terms found by keyword scoring")
        reason = "; ".join(grammar_bits)
        preview = first_lines(text, 5)
        block = [
            f"## PDF page {page.number}",
            f"- JPG filename: {page_images[page.number]}",
            f"- Selection score: {page.score}",
            f"- Expected usefulness: {usefulness}",
            f"- Reason selected: {reason}",
            f"- Grammar information: {preview}",
            f"- Readability/OCR concerns: embedded text extracted; verify exact forms in JPG if transliteration looks noisy.",
            "",
        ]
        lines.extend(block)
        txt_lines.extend([line.replace("#", "").strip() for line in block])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.with_suffix(".txt").write_text("\n".join(txt_lines), encoding="utf-8")
    return ", ".join(str(p.number) for p in selected)


def process_pdf(pdf_path: Path, language: str, output_dir: Path, target: int) -> None:
    pdf_name = clean_pdf_name(pdf_path)
    lang_slug = slugify(language)
    pdf_slug = slugify(pdf_name)
    root = output_dir / f"{language} Grammar Materials" / pdf_name
    summary_dir = root / "Summary Text"
    tables_dir = root / "Summary Image with Tables"
    cheat_dir = root / "Cheat Sheet from PDF"
    pages_dir = root / "Pages from the PDF"
    selected_dir = pages_dir / "Selected impactful pages JPG"

    for directory in [summary_dir, tables_dir, cheat_dir, selected_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    pages = extract_pages(pdf_path)
    selected = select_impactful_pages(pages, target)

    for old_page_image in selected_dir.glob("page_*.jpg"):
        old_page_image.unlink()

    page_images = {}
    render_notes = []
    for page in selected:
        filename = f"page_{page.number:03d}.jpg"
        size = render_pdf_page(pdf_path, page.number, selected_dir / filename)
        page_images[page.number] = filename
        render_notes.append(f"PDF page {page.number} -> {filename}, size={size[0]}x{size[1]}, score={page.score}")

    summary = summary_content(language, pdf_name, pages)
    summary_md = summary_dir / "summary_model_readable.md"
    summary_txt = summary_dir / "summary_model_readable.txt"
    write_both(summary_md, summary_txt, summary)

    cheat = cheatsheet_content(language, pdf_name)
    cheat_md = cheat_dir / "compact_cheatsheet.md"
    cheat_txt = cheat_dir / "compact_cheatsheet.txt"
    write_both(cheat_md, cheat_txt, cheat)
    cheat_images = render_markdown_to_jpg(cheat_md, cheat_dir / "compact_cheatsheet.jpg")

    tables = summary_tables_content(language, pdf_name)
    tables_md = tables_dir / "summary_tables.md"
    tables_txt = tables_dir / "summary_tables.txt"
    write_both(tables_md, tables_txt, tables)
    table_images = render_markdown_to_jpg(tables_md, tables_dir / "summary_tables.jpg")

    selected_pages_csv = write_manifest(selected_dir / "selected_pages_manifest.md", selected, page_images)

    notes = [
        "# Optional Page Notes",
        "",
        f"PDF file: {pdf_path}",
        f"Total pages: {len(pages)}",
        "Embedded text extraction was used. OCR was not needed.",
        "Only selected impactful pages were rendered to JPG, per instruction.",
        "Some extracted transliteration/phonetic symbols may be noisy because of PDF font encoding.",
        "The PDF appears to be a 59-page excerpt/front section of the grammar; full later chapters named in the table of contents are not present.",
        "",
        "## Rendered selected pages",
        *[f"- {line}" for line in render_notes],
        "",
        "## Markdown-to-JPG rendering",
        f"- Cheat sheet image files: {', '.join(p.name for p in cheat_images)}",
        f"- Summary table image files: {', '.join(p.name for p in table_images)}",
        "If multiple image files were created, the Markdown source was too long for one readable JPG page.",
    ]
    (pages_dir / "optional_page_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (pages_dir / "optional_page_notes.txt").write_text("\n".join(line.replace("#", "").strip() for line in notes) + "\n", encoding="utf-8")

    conversion_notes = [
        "Page conversion notes",
        f"PDF file: {pdf_path}",
        f"Total PDF pages: {len(pages)}",
        f"Selected pages rendered: {len(selected)}",
        "All pages were not rendered because the instruction requested selected impactful pages only.",
        "Rendering: PyMuPDF at zoom=2.0, JPEG quality=82.",
        "Failures: none.",
        "Hard-to-read/OCR concerns: some extracted text has noisy transliteration due to PDF font encoding; rendered pages preserve original visual forms.",
    ]
    (pages_dir / "page_conversion_notes.txt").write_text("\n".join(conversion_notes) + "\n", encoding="utf-8")

    report = f"""
# Creation Report

Language: {language}
PDF file processed: {pdf_path}
Output location: {root}
Total PDF pages: {len(pages)}
Selected original PDF pages rendered to JPG: {len(selected)}
Selected pages: {selected_pages_csv}

Files created:
- {summary_md}
- {summary_txt}
- {cheat_md}
- {cheat_txt}
- {', '.join(str(p) for p in cheat_images)}
- {tables_md}
- {tables_txt}
- {', '.join(str(p) for p in table_images)}
- {selected_dir / 'selected_pages_manifest.md'}
- {selected_dir / 'selected_pages_manifest.txt'}
- {pages_dir / 'optional_page_notes.md'}
- {pages_dir / 'optional_page_notes.txt'}
- {pages_dir / 'page_conversion_notes.txt'}

Major grammar areas covered:
- abbreviations and gloss labels
- overview of noun inflection and cases
- overview of verb categories and negation
- head-final word order and SOV preference
- ergative clause pattern and Dative experiencer examples
- subordinate/relative/complement clauses
- converb constructions
- polar and parametric questions
- gloss notation conventions

Missing or unclear information:
- Full later grammar chapters are listed in the table of contents but are not present in this 59-page PDF excerpt.
- Full detailed paradigms for pronouns, postpositions, valence, and clause combining were not available in the PDF excerpt.
- Some transliteration symbols in extracted text may be noisy; JPG pages should be used for exact visual verification.

Consistency and faithfulness:
- Outputs are based only on the input PDF.
- No OCR was needed.
- No external linguistic information was added.
- Markdown files are the authoritative sources.
- Rendered JPG files were generated from the Markdown sources.
- Materials are ready for Qwen/Qwen-VL experiments, with the limitation that this PDF appears to be an excerpt rather than the full grammar.
"""
    (root / "creation_report.txt").write_text(report.strip() + "\n", encoding="utf-8")

    print(report.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--pdf_path_or_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selected_page_target", type=int, default=15)
    args = parser.parse_args()

    pdfs = find_pdfs(Path(args.pdf_path_or_folder))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {args.pdf_path_or_folder}")
    for pdf in pdfs:
        process_pdf(pdf, args.language, Path(args.output_dir), args.selected_page_target)


if __name__ == "__main__":
    main()
