import re


def strip_newlines(text, newline="\\n"):
    text = re.sub(newline, " ", text)
    text = re.sub("  +", " ", text)
    return text
