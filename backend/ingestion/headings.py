"""Turns text_extraction.py's detected heading lines into sections, each
with a heading path (issue #26).

Font size is the only heading signal a PDF gives us, and in a Surgical
Technique brochure it fires on far more than headings: figure numbers
("1", "8a 9", "69c"), running footers ("4 MEDLINE UNITE"), doubled-font
text ("MEDLINE MEDLINE UNITE UNITE 3 3"), and titles wrapped across two
lines. Taken at face value those cut step text mid-sentence, drop the real
step heading that sat directly above a figure number, and leave three
"Step 1" chunks with nothing saying which technique each belongs to. This
module repairs that before chunking.py packs anything:

- A heading line with no word in it (a figure label) or that's a running
  footer is demoted back to body text, so its text stays where it was.
- Consecutive heading lines on the same page are joined ("Step 9. Akin
  Osteotomy" + "& Wire Placement"), unless the second is a step heading,
  which makes the first its parent. Before this, every heading but the
  last of such a run was silently dropped.
- Headings are cleaned: doubled-font repeats collapsed, a stray figure
  number trailing a step title moved to the body, letter-spaced fragments
  ("R emove", "1 st") re-joined where the document itself proves the
  joined word exists.
- Each section gets a heading path: `[technique, step]` for a step under a
  technique title, `[step]` for a step with no title above it, and
  `[heading]` for anything else. The document itself is the implicit root
  (every chunk already carries its document_id), so it isn't repeated in
  the path.

Only text the document itself contains ever goes into a path. Where a
technique title isn't in the extracted text at all (several Ankle Trauma
techniques print theirs as artwork), the steps get no parent rather than
an invented one.
"""
import re
from dataclasses import dataclass, field

HEADING_PREFIX = "# "

# "Step 3. ...", "Step 1 | ...", "Optional step 4 | ...", and a bare
# "Step 2". Case-insensitive: the Ankle brochure writes "Optional step".
_STEP_PATTERN = re.compile(r"^(?:optional\s+)?step\s*(\d+)\b", re.IGNORECASE)
# "Optional | Plate bending" sits between numbered steps and belongs to the
# same technique, but doesn't advance the numbering.
_OPTIONAL_PATTERN = re.compile(r"^optional\b", re.IGNORECASE)
# A token counts as a word once it has two letters in a row -- "8a", "69c",
# "S" and "ø16" don't, "Step" and "TN" do.
_WORD_PATTERN = re.compile(r"[^\W\d_]{2,}")
_FIGURE_WORDS = {"fig", "figs", "figure", "figures"}
# A label whose number is part of the heading, never a page number: these
# repeat on every page by design and must stay headings.
_NUMBERED_LABEL_PATTERN = re.compile(
    r"^(?:optional\s+)?(?:step|section|part|stage|phase|table)\s*\d", re.IGNORECASE
)
# A figure number printed on the same line as a step title, e.g. REFLEX's
# "Step 3. 4-Leg Staple Bone Preparation 4". Only stripped from step
# headings, where the step's own number is at the front; on any other
# heading a trailing number may be part of the name.
_TRAILING_FIGURE_PATTERN = re.compile(r"^(?P<heading>.*[^\W\d_]{2}.*?)\s+(?P<figure>\d{1,3}[a-z]?)$")
_ORDINAL_SPLIT_PATTERN = re.compile(r"\b(\d+) (st|nd|rd|th)\b")
_LETTER_SPLIT_PATTERN = re.compile(r"(?<![\w])([A-Z]) ([a-z]{2,})\b")
# A running header/footer recurs on at least this many pages.
_BOILERPLATE_MIN_PAGES = 3
# A technique title's own text, at most: REFLEX's titles carry only a
# figure number ("1") under them, Ankle's "continued" page one word.
_TITLE_ONLY_MAX_WORDS = 3
# A technique title may carry a short intro paragraph on the page where its
# first step starts. Anything longer (a product feature box, a table) is a
# section of its own, not a title over the steps that follow it.
_TITLE_INTRO_MAX_WORDS = 40

_Line = tuple[str, int | None]


@dataclass
class Section:
    heading_path: list[str]
    body_lines: list[_Line]


@dataclass
class _Block:
    heading: str | None
    page: int | None
    body_lines: list[_Line] = field(default_factory=list)


def step_number(heading: str) -> int | None:
    match = _STEP_PATTERN.match(heading)
    return int(match.group(1)) if match else None


def is_step_heading(heading: str) -> bool:
    """A numbered step or an "Optional ..." aside between steps. Exposed for
    the step-extraction work that builds on these paths (issue #28)."""
    return bool(_STEP_PATTERN.match(heading) or _OPTIONAL_PATTERN.match(heading))


def outline_sections(tagged_lines: list[_Line]) -> list[Section]:
    """Splits page-tagged lines into sections on their `# ` heading lines.
    Text with no heading line at all comes back as one section with an
    empty path, as before this module existed."""
    if not any(line.startswith(HEADING_PREFIX) for line, _ in tagged_lines):
        return [Section([], tagged_lines)]
    vocabulary = _vocabulary(tagged_lines)
    boilerplate = _boilerplate_cores(tagged_lines)
    blocks = _heading_blocks(tagged_lines, vocabulary, boilerplate)
    return _assign_heading_paths(blocks)


def repair_split_words(text: str, vocabulary: set[str]) -> str:
    """Re-joins "1 st" and letter-spaced fragments like "R emove". A
    capital letter is joined to the lowercase run after it only when the
    joined word appears elsewhere in the same document, so "A burr" and
    "C arm" are left alone."""
    text = _ORDINAL_SPLIT_PATTERN.sub(r"\1\2", text)

    def join_if_known(match: re.Match) -> str:
        joined = match.group(1) + match.group(2)
        return joined if joined.casefold() in vocabulary else match.group(0)

    return _LETTER_SPLIT_PATTERN.sub(join_if_known, text)


def collapse_repeated_tokens(text: str) -> str:
    """Doubled-font text extracts with every word repeated in place:
    "MEDLINE MEDLINE UNITE UNITE 3 3" -> "MEDLINE UNITE 3". Only collapses
    when *every* token repeats the same number of times."""
    tokens = text.split()
    for repeat in range(len(tokens), 1, -1):
        if len(tokens) % repeat:
            continue
        groups = [tokens[start : start + repeat] for start in range(0, len(tokens), repeat)]
        if all(len(set(group)) == 1 for group in groups):
            return " ".join(group[0] for group in groups)
    return text


def _vocabulary(tagged_lines: list[_Line]) -> set[str]:
    return {
        word.casefold()
        for line, _ in tagged_lines
        for word in re.findall(r"[A-Za-z]{3,}", line)
    }


def _letters_core(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def _boilerplate_cores(tagged_lines: list[_Line]) -> set[str]:
    """The letters of every line that recurs on several pages, page number
    aside: "4 MEDLINE UNITE", "MEDLINE UNITE 9", "medlineunite.com 11"."""
    pages_by_core: dict[str, set[int]] = {}
    for line, page in tagged_lines:
        if page is None:
            continue
        core = _letters_core(collapse_repeated_tokens(line.removeprefix(HEADING_PREFIX)))
        if len(core) >= 4:
            pages_by_core.setdefault(core, set()).add(page)
    return {core for core, pages in pages_by_core.items() if len(pages) >= _BOILERPLATE_MIN_PAGES}


def _is_figure_label(heading: str) -> bool:
    """"1", "8a 9", "69c", "S L", "Fig. 3": nothing a reader would call a
    heading."""
    return all(
        not _WORD_PATTERN.search(token) or token.casefold().rstrip(".") in _FIGURE_WORDS
        for token in heading.split()
    )


def _is_running_boilerplate(heading: str, boilerplate: set[str]) -> bool:
    if _NUMBERED_LABEL_PATTERN.match(heading) or not re.search(r"\d", heading):
        return False
    return _letters_core(heading) in boilerplate


def _clean_heading(raw: str, vocabulary: set[str]) -> tuple[str, str | None]:
    """The cleaned heading, and a trailing figure number split off it (or
    None) for the caller to keep with the body."""
    heading = repair_split_words(collapse_repeated_tokens(raw.strip()), vocabulary)
    figure = None
    if _STEP_PATTERN.match(heading):
        match = _TRAILING_FIGURE_PATTERN.match(heading)
        if match and len(heading.split()) > 2:
            heading, figure = match.group("heading"), match.group("figure")
    return heading.rstrip(" ,;:|"), figure


def _has_text(lines: list[_Line]) -> bool:
    return any(line.strip() for line, _ in lines)


def _word_count(lines: list[_Line]) -> int:
    return sum(1 for line, _ in lines for token in line.split() if _WORD_PATTERN.search(token))


def _heading_blocks(
    tagged_lines: list[_Line], vocabulary: set[str], boilerplate: set[str]
) -> list[_Block]:
    blocks: list[_Block] = []
    current = _Block(heading=None, page=None)
    for line, page in tagged_lines:
        if not line.startswith(HEADING_PREFIX):
            current.body_lines.append((repair_split_words(line, vocabulary), page))
            continue
        raw = line[len(HEADING_PREFIX) :]
        heading, figure = _clean_heading(raw, vocabulary)
        if not heading or _is_figure_label(heading) or _is_running_boilerplate(heading, boilerplate):
            current.body_lines.append((repair_split_words(raw, vocabulary), page))
            continue
        joins_previous = (
            current.heading is not None
            and current.page == page
            and not _has_text(current.body_lines)
            and not is_step_heading(heading)
        )
        if joins_previous:
            current.heading = f"{current.heading} {heading}"
        else:
            blocks.append(current)
            current = _Block(heading=heading, page=page)
        if figure:
            current.body_lines.append((figure, page))
    blocks.append(current)
    # The leading block before the first heading is only worth keeping if
    # it holds text -- otherwise it's the empty starting block.
    return [block for block in blocks if block.heading is not None or _has_text(block.body_lines)]


def _titles_its_steps(block: _Block, next_block: _Block | None) -> bool:
    """Whether a non-step heading is the title of the steps right after it:
    the next heading is a step, and the title has at most a few words
    under it (a figure number, "continued") or a short intro on the same
    page as that step."""
    if next_block is None or next_block.heading is None or not is_step_heading(next_block.heading):
        return False
    words = _word_count(block.body_lines)
    if words <= _TITLE_ONLY_MAX_WORDS:
        return True
    return words <= _TITLE_INTRO_MAX_WORDS and block.page == next_block.page


def _continues_steps(next_block: _Block | None, last_step: int | None) -> bool:
    """Whether the next heading carries on the current technique's steps
    (a later step number, or an unnumbered "Optional" aside)."""
    if next_block is None or next_block.heading is None or not is_step_heading(next_block.heading):
        return False
    number = step_number(next_block.heading)
    return number is None or last_step is None or number > last_step


def _assign_heading_paths(blocks: list[_Block]) -> list[Section]:
    sections: list[Section] = []
    technique: str | None = None
    last_step: int | None = None
    carried: list[_Line] = []
    for position, block in enumerate(blocks):
        next_block = blocks[position + 1] if position + 1 < len(blocks) else None
        body_lines = carried + block.body_lines
        carried = []
        heading = block.heading
        if heading is None:
            path: list[str] = []
        elif is_step_heading(heading):
            number = step_number(heading)
            if number is not None and last_step is not None and number <= last_step:
                # Numbering restarted with no title of its own above it:
                # a new technique whose name we don't have.
                technique = None
            path = [technique, heading] if technique else [heading]
            if number is not None:
                last_step = number
        elif _titles_its_steps(block, next_block):
            technique, last_step = heading, None
            path = [heading]
            if _word_count(block.body_lines) <= _TITLE_ONLY_MAX_WORDS:
                # A title with only a figure number or "continued" under it
                # isn't worth a chunk of its own: its text moves into the
                # first step's chunk, which now carries the title in its
                # path anyway.
                carried = body_lines
                continue
        elif technique is not None and _continues_steps(next_block, last_step):
            path = [technique, heading]
        else:
            technique, last_step = None, None
            path = [heading]
        sections.append(Section(path, body_lines))
    return sections
