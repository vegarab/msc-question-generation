import re


def process_text(context, answer, question):
    source_text = f"answer: {answer} context: {context}"
    target_text = f"{question}"
    return {'source_text': source_text, 'target_text': target_text}


def strip_newlines(text, newline="\\n"):
    text = re.sub(newline, " ", text)
    text = re.sub("  +", " ", text)
    return text
