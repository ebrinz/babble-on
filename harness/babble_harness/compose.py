"""The device writes the brief: 16 entropy bytes pick a subject, setting,
mood and form from fixed lists ordered roughly from ordinary to strange, and
the prompt is built from them. Uses the bytes -> uniform -> index contract in
`noise.py`, so a byte stream skewed high picks from the strange end of every
list. Promoted from spikes/prompt_compose (2026-10-14)."""
from __future__ import annotations

from .noise import bytes_to_uniforms, uniform_to_index

SLOTS: dict[str, list[str]] = {
    "subject": [
        "a cup of tea", "a dog", "a garden", "a bus stop", "an old chair", "a kitchen", "a bicycle", "a letter",
        "a market", "a schoolteacher", "a fisherman", "a bakery", "a violin", "a lighthouse keeper",
        "a beekeeper", "a night nurse", "a clockmaker", "a cartographer", "a beached whale", "a ghost train",
        "an abandoned observatory", "a swarm of moths", "a sentient fog", "a lighthouse with no sea",
        "a library of unwritten books", "a man made of keys", "a staircase that counts", "a moon that hums",
        "a city of glass bees", "a borrowed shadow", "the last word of a dead language", "a door into weather",
    ],
    "setting": [
        "on a quiet morning", "in a small town", "in a busy café", "at a train station", "in a suburban house",
        "on a farm", "in a hospital corridor", "at the seaside", "in a snowed-in cabin", "in a flooded basement",
        "on a night ferry", "in a closed museum", "at the edge of a desert", "inside a lighthouse",
        "on an oil rig", "in a cathedral at dawn", "beneath a frozen lake", "in a city after the evacuation",
        "on a generation ship", "inside a whale", "in the space between two clocks", "in a house that rearranges itself",
        "on the far side of a mirror", "in a forest of radio towers", "inside a dream someone else is having",
        "on a planet with three suns", "at the bottom of a well of stars", "in a museum of lost sounds",
        "inside a single raindrop", "where the map runs out", "in the year that never happened", "nowhere at all",
    ],
    "mood": [
        "calm", "warm", "cheerful", "nostalgic", "wistful", "tender", "restless", "hopeful", "bittersweet",
        "lonely", "uneasy", "playful", "solemn", "melancholy", "tense", "eerie", "feverish", "giddy",
        "grief-stricken", "furious", "dreamlike", "uncanny", "ecstatic", "menacing", "hallucinatory",
        "absurd", "sacred", "unhinged", "weightless", "apocalyptic", "rapturous", "unspeakable",
    ],
    "form": [
        "a short scene", "a diary entry", "a letter", "a short story opening", "a description", "a memory",
        "a postcard", "a recipe", "a weather report", "a eulogy", "a confession", "a lullaby in prose",
        "a field guide entry", "a ship's log", "a museum placard", "an instruction manual", "a prayer",
        "a classified advertisement", "a police report", "a dream journal", "a list of rules", "a radio broadcast",
        "an encyclopedia entry", "a ransom note", "a translation of birdsong", "a map legend",
        "a message in a bottle", "a horoscope", "a court transcript", "a spell", "a footnote to nothing",
        "the last page of a novel",
    ],
}
BYTES_PER_BRIEF = 4 * len(SLOTS)


def compose(raw: bytes) -> tuple[str, dict[str, str]]:
    """(prompt, picks) from the first BYTES_PER_BRIEF bytes of `raw`."""
    if len(raw) < BYTES_PER_BRIEF:
        raise ValueError(f"brief needs {BYTES_PER_BRIEF} bytes, got {len(raw)}")
    us = bytes_to_uniforms(raw[:BYTES_PER_BRIEF], len(SLOTS))
    picks = {name: items[int(uniform_to_index(float(u), len(items)))]
             for (name, items), u in zip(SLOTS.items(), us)}
    prompt = (f"Write {picks['form']} about {picks['subject']}, {picks['setting']}. "
              f"The mood is {picks['mood']}. Keep it under 200 words.")
    return prompt, picks
