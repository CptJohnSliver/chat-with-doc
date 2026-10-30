"""Text utilities for document_process."""

import re

_LINK = re.compile(r"(?<!!)\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)]+)\)", re.DOTALL)


def sanitize_md_links(text: str) -> str:
    out = _LINK.sub(lambda m: m.group(1), text)
    return re.sub(r"\[\[([^\[\]]+)\]\]", r"[\1]", out)
