import spacy
import jsonlines
import json
import zipfile
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from loguru import logger
import tempfile

from prepare_utils import (
    clean_white_spaces,
    separate_entities_by_label,
    filter_overlaping_entities,
    convert_to_conll,
)

SPACY_MODEL = "fr_core_news_sm"
ANNOTATION_DIR = "m2annotations"
OUTPUT_DIR = "corpusCasM2"
SPLIT_SEEDS = (67, 33)
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

try:
    nlp = spacy.load(SPACY_MODEL)
except:
    spacy.cli.download(SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL)

with tempfile.TemporaryDirectory() as tmpdir:
    # unzip annotation files
    zip_list = glob.glob(f"{ANNOTATION_DIR}/*.zip")
    logger.info(f"identified {len(zip_list)} zip files from '{ANNOTATION_DIR}/'")
    i = 0
    for zip_file in zip_list:
        with zipfile.ZipFile(zip_file, "r") as archive:
            archive.extract("all.jsonl", f"{tmpdir}/annot_{i}")
            i += 1

    # create spacy docs from jsonl files
    dir_list = glob.glob(f"{tmpdir}/*")
    logger.info(f"identified {len(dir_list)} annotation files from '{tmpdir}/'")
    docs = []
    for dirpath in tqdm(dir_list):
        with jsonlines.open(f"{dirpath}/all.jsonl", mode="r") as reader:
            for doc in reader:
                entities = []
                for ent in doc["entities"]:
                    entities.append(
                        (ent["start_offset"], ent["end_offset"], ent["label"])
                    )
                docs.append(
                    {
                        "id": doc["id"],
                        "doc": nlp(doc["text"]),
                        "text": doc["text"],
                        "entities": entities,
                    }
                )

logger.info(f"loaded {len(docs)} docs")

# preprocess annotations
for doc in docs:
    doc = clean_white_spaces(doc)
    doc = separate_entities_by_label(doc, "scope", "scopes")
    doc["entities"] = filter_overlaping_entities(doc["entities"])

# create conlls from annotations
docs_in_error = []
for i, doc in enumerate(docs):
    try:
        doc["conll"] = convert_to_conll(doc, return_sequences=True)
    except Exception as e:
        logger.warning(e)
        docs_in_error.append(i)
logger.info(f"{len(docs_in_error)}/{len(docs)} docs in error after conll conversion")

docs = [doc for i, doc in enumerate(docs) if i not in docs_in_error]

logger.info(f"final number of docs: {len(docs)} ")

# split datasets
conlls = [doc["conll"] for doc in docs]
train_set, test_set = train_test_split(
    conlls, random_state=SPLIT_SEEDS[0], test_size=TEST_SIZE
)
train_set, validation_set = train_test_split(
    train_set, random_state=SPLIT_SEEDS[1], test_size=VALIDATION_SIZE
)


# write datasets
train_set_concat = pd.concat(train_set)
test_set_concat = pd.concat(test_set)
validation_set_concat = pd.concat(validation_set)

logger.info(
    f"train_set: {len(train_set)} documents, {len(train_set_concat.index)} sentences"
)
logger.info(
    f"test_set: {len(test_set)} documents, {len(test_set_concat.index)} sentences"
)
logger.info(
    f"validation_set: {len(validation_set)} documents, {len(validation_set_concat.index)} sentences"
)

if not os.path.exists(f"{OUTPUT_DIR}"):
    logger.info(f"create '{OUTPUT_DIR}'.")
    os.makedirs(f"{OUTPUT_DIR}")

with open(f"{OUTPUT_DIR}/train.json", "w") as f:
    json.dump(train_set_concat.to_dict(orient="records"), f)
with open(f"{OUTPUT_DIR}/test.json", "w") as f:
    json.dump(test_set_concat.to_dict(orient="records"), f)
with open(f"{OUTPUT_DIR}/validation.json", "w") as f:
    json.dump(validation_set_concat.to_dict(orient="records"), f)

logger.info(f"written train.json, test.json and validation.json to {OUTPUT_DIR}")
