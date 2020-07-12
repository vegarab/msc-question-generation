# coding=utf-8
# Lint as: python3
"""MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of
Text"""
from __future__ import absolute_import, division, print_function

import csv
import json
import os

import nlp


_CITATION = """\
@inproceedings{mctest,
    title = "{MCT}est: A Challenge Dataset for the Open-Domain Machine Comprehension of Text",
    author = "Richardson, Matthew  and
      Burges, Christopher J.C.  and
      Renshaw, Erin",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1020",
    pages = "193--203",
}
"""

_DESCRIPTION = """\
We present MCTest, a freely available set of stories and associated questions
intended for research on the machine comprehension of text. Previous work on
machine comprehension (e.g., semantic modeling) has made great strides, but
primarily focuses either on limited-domain datasets, or on solving a more
restricted goal (e.g., open-domain relation extraction). In contrast, MCTest
requires machines to answer multiple-choice reading comprehension questions
about fictional stories, directly tackling the high-level goal of open-domain
machine comprehension. Reading comprehension can test advanced abilities such
as causal reasoning and understanding the world, yet, by being multiple-choice,
still provide a clear metric. By being fictional, the answer typically can be
found only in the story itself. The stories and questions are also carefully
limited to those a young child would understand, reducing the world knowledge
that is required for the task. We present the scalable crowd-sourcing methods
that allow us to cheaply construct a dataset of 500 stories and 2000 questions.
By screening workers (with grammar tests) and stories (with grading), we have
ensured that the data is the same quality as another set that we manually
edited, but at one tenth the editing cost. By being open-domain, yet carefully
restricted, we hope MCTest will serve to encourage research and provide a clear
metric for advancement on the machine comprehension of text.
"""

_URL = ""
_TEST_FILE = ""
_TRAIN_FILE = ""
_DEV_FILE = ""


class MCTest(nlp.GeneratorBasedBuilder):
    VERSION = nlp.Version("0.1.0")

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "id": nlp.Value("int32"),
                    "context": nlp.Value("string"),
                    "question": nlp.Value("string"),
                    "answer": nlp.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="https://mattr1.github.io/mctest/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = self.config.data_files
        dl_dir = dl_manager.download_and_extract(urls_to_download)
        return [
            nlp.SplitGenerator(
                name=nlp.Split.TRAIN,
                gen_kwargs={"filepath": dl_dir["train"], "split": "train"},
            ),
            nlp.SplitGenerator(
                name=nlp.Split.TEST,
                gen_kwargs={"filepath": dl_dir["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath) as f:
            data = csv.DictReader(f)
            for id_, row in enumerate(data):
                yield id_, {
                    "id": id_,
                    "context": row["context"],
                    "question": row["question"],
                    "answer": row["answer"]
                }
