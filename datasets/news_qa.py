# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""NewsQA: A Machine Comprehension Dataset"""

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import nlp


_CITATION = """\
@article{Trischler_2017,
   title={NewsQA: A Machine Comprehension Dataset},
   url={http://dx.doi.org/10.18653/v1/W17-2623},
   DOI={10.18653/v1/w17-2623},
   journal={Proceedings of the 2nd Workshop on Representation Learning for NLP},
   publisher={Association for Computational Linguistics},
   author={Trischler, Adam and \
           Wang, Tong and \
           Yuan, Xingdi and \
           Harris, Justin and \
           Sordoni, Alessandro and \
           Bachman, Philip and \
           Suleman, Kaheer},
   year={2017}
}
"""

_DESCRIPTION = """\
We present NewsQA, a challenging machine comprehension dataset of over 100,000
human-generated question-answer pairs. Crowdworkers supply questions and
answers based on a set of over 10,000 news articles from CNN, with answers
consisting of spans of text from the corresponding articles. We collect this
dataset through a four-stage process designed to solicit exploratory questions
that require reasoning. A thorough analysis confirms that NewsQA demands
abilities beyond simple word matching and recognizing textual entailment. We
measure human performance on the dataset and compare it to several strong
neural models. The performance gap between humans and machines (0.198 in F1)
indicates that significant progress can be made on NewsQA through future
research.  """


class NewsQAConfig(nlp.BuilderConfig):
    """BuilderConfig for NewsQA."""

    def __init__(self, **kwargs):
        """BuilderConfig for NewsQA.

    Args:
      **kwargs: keyword arguments forwarded to super.
    """
        super(NewsQAConfig, self).__init__(**kwargs)


class NewsQA(nlp.GeneratorBasedBuilder):
    """NewsQA: A Machine Comprehension Dataset"""

    _URL = "https://multiqa.s3.amazonaws.com/squad2-0_format_data/"
    _DEV_FILE = "NewsQA_dev.json.gz"
    _TRAINING_FILE = "NewsQA_train.json.gz"

    BUILDER_CONFIGS = [
        NewsQAConfig(
            name="plain_text",
            version=nlp.Version(
                "1.0.0", "New split API (https://tensorflow.org/datasets/splits)"),
            description="Plain text",
        ),
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "id": nlp.Value("string"),
                    "title": nlp.Value("string"),
                    "context": nlp.Value("string"),
                    "question": nlp.Value("string"),
                    "answers": nlp.features.Sequence(
                        {"text": nlp.Value("string"),
                         "answer_start": nlp.Value("int32"), }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://datasets.maluuba.com/NewsQA",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": os.path.join(self._URL, self._TRAINING_FILE),
            "test": os.path.join(self._URL, self._DEV_FILE),
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={
                               "filepath": downloaded_files["train"]}),
            nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={
                               "filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        with open(filepath) as f:
            news_qa = json.load(f)
            for article in news_qa["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"]
                                         for answer in qa["answers"]]
                        answers = [answer["text"].strip()
                                   for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {"answer_start": answer_starts, "text": answers, },
                        }
