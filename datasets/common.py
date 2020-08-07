import re


def strip_newlines(text, newline="\\n"):
    text = re.sub(newline, " ", text)
    text = re.sub("  +", " ", text)
    return text


def create_dict(context, answer, question):
    return {
        "context": context,
        "answer": answer,
        "question": question
    }
