import itertools
from spacy.training import offsets_to_biluo_tags, tags_to_entities
import warnings
from spacy.tokens import Span, Token
import pandas as pd


def clean_white_spaces(doc):
    new_ents = []
    for ent in doc["entities"]:
        span = doc["text"][ent[0] : ent[1]]
        trailing_spaces = len(span) - len(span.rstrip())
        leading_spaces = len(span) - len(span.lstrip())
        new_ents.append((ent[0] + leading_spaces, ent[1] - trailing_spaces, ent[2]))
    doc["entities"] = new_ents
    return doc


def separate_entities_by_label(doc, label, out_key):
    new_ents = []
    out_keys = []
    for ent in doc["entities"]:
        if ent[2] == label:
            out_keys.append(ent)
        else:
            new_ents.append(ent)
    doc["entities"] = new_ents
    doc[out_key] = out_keys
    return doc


def filter_overlaping_entities(entities):
    to_exclude = []
    for ent1, ent2 in itertools.combinations(entities, 2):
        span1 = range(ent1[0], ent1[1])
        span2 = range(ent2[0], ent2[1])
        if len(set(span1).intersection(span2)) > 0:
            to_exclude.append(ent2)
    return [ent for ent in entities if ent not in to_exclude]


def extract_tokens_with_offsets(doc):
    tokens = []
    for i, sent in enumerate(doc["doc"].sents):
        for token in sent:
            tokens.append(
                dict(
                    doc_id=doc["id"],
                    sent_id=f"{doc['id']}_{i:03d}",
                    token=token.text,
                    start=token.idx,
                    end=token.idx + len(token.text),
                    pos=token.pos_,
                )
            )
    return tokens


def biluo_tags_to_entities(doc, tags):
    entities = tags_to_entities(tags)
    spans = []
    for label, start, end in entities:
        spans.append(Span(doc, start, end + 1, label=label))
    return spans


def convert_to_conll(doc, return_sequences=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        biluo = offsets_to_biluo_tags(doc["doc"], doc["entities"])
    doc["doc"].ents = biluo_tags_to_entities(doc["doc"], biluo)
    tokens = extract_tokens_with_offsets(doc)
    conll = pd.DataFrame(tokens)
    conll["biluo"] = biluo
    conll["bio"] = biluo
    conll["bio"].replace("L-", "I-", regex=True, inplace=True)
    conll["bio"].replace("U-", "B-", regex=True, inplace=True)
    conll["bio"].replace("^-$", "0", regex=True, inplace=True)

    if return_sequences:
        conll = (
            conll.groupby(["doc_id", "sent_id"])
            .agg(
                {
                    "token": lambda x: list(x),
                    "biluo": lambda x: list(x),
                    "bio": lambda x: list(x),
                    "pos": lambda x: list(x),
                }
            )
            .reset_index()
        )

    return conll
