import re, pandas as pd

V = set("AEIOUY")
WORDS = {"CAT", "GAP", "KEY", "BALL", "LOW", "DISH", "FAST", "LUV", "EAT", "HOG", "BID", "TAP", "SAVE", "WEN", "BEN",
         "CAKE", "COST", "PEP", "SEE", "HUM", "MAT", "GIS", "MO", "BIG", "AIR", "PLAY", "ZION", "HAS", "FUN", "COKE",
         "NICE", "WING", "LOVE", "HD", "MOS", "EL", "ED", "TEL", "IT", "ON", "ALL", "ARE", "FOX", "NOW", "DAY", "TAN"}

def pronounceable(tk: str) -> int:
    t = tk.split(".")[0].upper()
    if t in WORDS:
        return 1
    if len(t) < 3 or not any(c in V for c in t):
        return 0
    # consonant/vowel pattern: no run of 3+ consonants, no double-vowel-only strings
    pat = "".join("V" if c in V else "C" for c in t)
    return int("CCC" not in pat and re.fullmatch(r"C?V(C|CV|VC|CC)?(V|C)?", pat) is not None)

def signal(ctx, t):
    m = ctx.lk[(ctx.lk.date == t) & (ctx.lk.permno > 0)].drop_duplicates("permno").set_index("permno").ticker
    return m.map(pronounceable).astype(float)
