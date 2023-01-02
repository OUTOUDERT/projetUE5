import datasets
import json

_DESCRIPTION = """\
corpusCas is a dataset of clinical cases reanotated from the CORPUS CAS by masters students from M2 IBM/DMS. 
"""
_HOMEPAGE_URL = "https://github.com/aneuraz/corpusCasM2"
_DATA_URLS = {
    "train": "./train.json",
    "test": "./test.json",
    "validation": "./validation.json",
}


class corpusCasM2(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    #    "doc_id": datasets.Value("int8"),
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "B-date",
                                "I-date",
                                "B-duration",
                                "I-duration",
                                "B-problem",
                                "I-problem",
                                "B-treatment",
                                "I-treatment",
                                "B-test",
                                "I-test",
                                "B-frequency",
                                "I-frequency",
                                "O",
                            ]
                        )
                    ),
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
            # data_urls = _DATA_URLS
        )

    def _split_generators(self, dl_manager):

        downloaded_files = dl_manager.download_and_extract(_DATA_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
        ]

    def _generate_examples(self, filepath):
        sentence_counter = 0
        with open(filepath, encoding="utf-8") as f:
            dt = json.load(f)

            for sentence in dt:
                id_ = sentence["sent_id"]

                yield id_, {
                    "id": id_,
                    "tokens": sentence["token"],
                    "ner_tags": sentence["bio"],
                }
