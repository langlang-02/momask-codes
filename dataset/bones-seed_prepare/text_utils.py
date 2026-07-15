"""Text conversion helpers for MoMask text files."""

from __future__ import annotations

import re


_VERBS = {
    "walk", "walks", "walking", "run", "runs", "running", "jog", "jogs", "jump", "jumps",
    "turn", "turns", "stand", "stands", "sit", "sits", "dance", "dances", "reach",
    "reaches", "pick", "picks", "hold", "holds", "lift", "lifts", "lower", "lowers",
    "wave", "waves", "bend", "bends", "throw", "throws", "kick", "kicks", "climb",
    "climbs", "crawl", "crawls", "rotate", "rotates", "open", "opens", "close",
    "closes", "touch", "touches", "look", "looks", "point", "points",
}

_LOC = {
    "left", "right", "forward", "backward", "back", "up", "down", "around", "sideways",
    "clockwise", "counterclockwise",
}

_BODY = {
    "hand", "hands", "arm", "arms", "leg", "legs", "foot", "feet", "head", "hips",
    "knee", "knees", "shoulder", "shoulders", "body", "torso",
}


def clean_caption(text: str | None) -> str | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return re.sub(r"\s+", " ", text)


def token_with_pos(word: str) -> str:
    token = word.lower()
    if token in _VERBS:
        pos = "VERB"
    elif token in _LOC:
        pos = "Loc_VIP"
    elif token in _BODY:
        pos = "Body_VIP"
    else:
        pos = "OTHER"
    return f"{token}/{pos}"


def caption_to_tokens(caption: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\\d+", caption)
    if not words:
        return "unk/OTHER"
    return " ".join(token_with_pos(word) for word in words)


def format_momask_text_line(caption: str, start: float = 0.0, end: float = 0.0) -> str:
    return f"{caption}#{caption_to_tokens(caption)}#{start:.2f}#{end:.2f}"


def captions_from_metadata(row: dict[str, str], max_captions: int = 4) -> list[str]:
    columns = [
        "content_natural_desc_1",
        "content_natural_desc_2",
        "content_natural_desc_3",
        "content_natural_desc_4",
        "content_short_description",
        "content_short_description_2",
        "content_technical_description",
    ]
    captions = []
    seen = set()
    for col in columns:
        caption = clean_caption(row.get(col))
        if caption and caption.lower() not in seen:
            captions.append(caption)
            seen.add(caption.lower())
        if len(captions) >= max_captions:
            break
    return captions
