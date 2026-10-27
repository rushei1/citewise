import logging
import re
from dataclasses import dataclass, field

from app.schemas import CLAIM_FIELDS, Paper
from app.services.llm import LLMCache, LLMClient, Message
from app.services.summarize import PROMPT_VERSION

log = logging.getLogger(__name__)

THEME_VERSION = f"{PROMPT_VERSION}/theme"  # bumping PROMPT_VERSION invalidates both caches
MIN_PAPERS = 2  # a "theme" of one paper is just that paper's summary

SYSTEM_PROMPT = """You write the theme summary of a group of related papers for a literature \
review. Write 3 to 5 sentences. Use ONLY the numbered papers and claims you are given; add no \
other facts. After each statement, cite the supporting papers by number, one number per \
bracket, like [1] or [2][3]. Only use numbers that appear in the list. Plain prose, no \
headings or lists."""

_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_NUMBERS = re.compile(r"\s*\d+(\s*,\s*\d+)*\s*")


@dataclass
class Theme:
    text: str | None  # None: rejected because no valid citation survived
    invalid_citations: list[int] = field(default_factory=list)


def check_citations(text: str, n_papers: int) -> tuple[str, list[int], int]:
    """Strip [n] citations that do not name a paper 1..n_papers.

    Returns (cleaned text, invalid numbers, count of valid citations kept). Brackets that are
    not numeric citations are left alone."""
    invalid: list[int] = []
    valid = 0

    def fix(m: re.Match) -> str:
        nonlocal valid
        if not _NUMBERS.fullmatch(m.group(1)):
            return m.group(0)
        nums = [int(x) for x in m.group(1).split(",")]
        good = [n for n in nums if 1 <= n <= n_papers]
        invalid.extend(n for n in nums if not 1 <= n <= n_papers)
        valid += len(good)
        return f"[{', '.join(map(str, good))}]" if good else ""

    cleaned = _BRACKET.sub(fix, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", re.sub(r"\s+([.,;:])", r"\1", cleaned)).strip()
    return cleaned, invalid, valid


def _claims_block(papers: list[Paper]) -> tuple[str, int]:
    """The numbered claim list the model may draw on, and how many claims it holds."""
    lines, total = [], 0
    for n, paper in enumerate(papers, 1):
        lines.append(f"[{n}] {paper.title}")
        s = paper.structured
        for name in CLAIM_FIELDS:
            for c in getattr(s, name, []) if s else []:
                lines.append(f"  - {name}: {c.text}")
                total += 1
    return "\n".join(lines), total


class ThemeWriter:
    """Writes a short cited theme summary from a cluster's verified claims only."""

    def __init__(self, client: LLMClient, cache: LLMCache | None = None):
        self.client = client
        self.cache = cache

    def write(self, papers: list[Paper]) -> Theme | None:
        """None when there is nothing to synthesize from. [n] is the 1-based position of a paper
        in `papers`. Raises LLMError if the call fails."""
        if len(papers) < MIN_PAPERS:
            return None
        block, total = _claims_block(papers)
        if total == 0:
            return None
        reply = self._complete(block)
        text, invalid, valid = check_citations(reply, len(papers))
        if invalid:
            log.warning("theme summary cited papers that are not in the cluster: %s", invalid)
        return Theme(text if valid else None, invalid)

    def _complete(self, block: str) -> str:
        key = self.cache.key(self.client.model, THEME_VERSION, block) if self.cache else None
        if key and (hit := self.cache.get(key)):
            return hit
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Papers and verified claims:\n{block}"},
        ]
        reply = self.client.complete(messages).strip()
        if key and reply:
            self.cache.put(key, reply)
        return reply
