"""Build docs/report.md from docs/report_template.md.

Every `{{include: FILE # HEADING}}` in the template is replaced by the body of that section of a
generated results file (paths are relative to eval/), so the report's numbers are copied, not
retyped. A missing file or heading is an error rather than an empty gap.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from eval.common import ROOT

TEMPLATE = ROOT / "docs" / "report_template.md"
OUT = ROOT / "docs" / "report.md"
EVAL_DIR = ROOT / "eval"
INCLUDE = re.compile(r"\{\{include:\s*(?P<file>[^#}]+?)\s*#\s*(?P<heading>[^}]+?)\s*\}\}")


def section(text: str, heading: str) -> str:
    """Body of the first '## heading' / '### heading' section (up to the next heading of the
    same or higher level), with its own sub-headings demoted one level."""
    lines = text.splitlines()
    heads = [(i, re.match(r"(#+)\s+(.*)", line)) for i, line in enumerate(lines)]
    found = next(((i, m) for i, m in heads if m and m.group(2).strip().startswith(heading)), None)
    if found is None:
        raise KeyError(f"heading {heading!r} not found")
    start, level = found[0], len(found[1].group(1))
    body = []
    for line in lines[start + 1 :]:
        m = re.match(r"(#+)\s", line)
        if m and len(m.group(1)) <= level:
            break
        body.append(f"#{line}" if m else line)
    return "\n".join(body).strip()


def render(template: str, base: Path = EVAL_DIR) -> str:
    def sub(m: re.Match) -> str:
        path = base / m.group("file")
        if not path.exists():
            raise FileNotFoundError(f"{path} (run `make eval` first)")
        return section(path.read_text(encoding="utf-8"), m.group("heading"))

    return INCLUDE.sub(sub, template)


def to_pdf(md: Path) -> str:
    """Convert with pandoc if it is installed; otherwise say so instead of failing."""
    if shutil.which("pandoc") is None:
        return "skipped: pandoc is not installed"
    pdf = md.with_suffix(".pdf")
    done = subprocess.run(
        ["pandoc", str(md), "-o", str(pdf), "--toc"], capture_output=True, text=True
    )
    return f"wrote {pdf}" if done.returncode == 0 else f"pandoc failed: {done.stderr[-500:]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdf", action="store_true", help="also convert to PDF if pandoc exists")
    args = ap.parse_args()
    OUT.write_text(render(TEMPLATE.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {OUT}")
    if args.pdf:
        print(to_pdf(OUT))


if __name__ == "__main__":
    main()
