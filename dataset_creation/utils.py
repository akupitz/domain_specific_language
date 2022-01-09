import glob
import os
import re
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup

from dataset_creation.config import BEFORE_TEXT_CONTEXT_SIZE, AFTER_TEXT_CONTEXT_SIZE

START_CHAR_END_CHAR_EXTRACTION_REGEX = re.compile("[\w\W]*char=(\d+),(\d+)")
COMMITTEE_REGEX = re.compile("ועדת" + "[\u0590-\u05FF ]*")
PROTOCOL_NUMBER_REGEX_FROM_TEXT = re.compile("פרוטוקול מס'" + "[ ]*" + "([\d]+)")
PROTOCOL_NUMBER_REGEX_FROM_DIR_NAME = re.compile("פרוטוקול_" + "([\d]+)")


@dataclass
class CatmaAnnotation:
    segment_id: str
    start_char: int
    end_char: int


def catma_annotations_to_df(catma_annotations):
    segment_ids = [catma_annotation.segment_id for catma_annotation in catma_annotations]
    start_chars = [catma_annotation.start_char for catma_annotation in catma_annotations]
    end_chars = [catma_annotation.end_char for catma_annotation in catma_annotations]
    return pd.DataFrame({"text_segment_catma_id": segment_ids, "start_char": start_chars, "end_char": end_chars})


def create_dataframe(protocol_dir):
    dataframes = []
    xml_paths = glob.glob(os.path.join(protocol_dir, "annotationcollections", "*.xml"))
    for xml_path in xml_paths:
        # note that sometimes we have multiple xml_paths, maybe we don't want multiple annotations for the same file
        if not os.path.exists(xml_path):
            print(f"xml path: {xml_path} does not exist")
            raise NotImplemented
        txt_paths = glob.glob(os.path.join(protocol_dir, "*.txt"))
        assert len(txt_paths) == 1
        txt_path = txt_paths[0]
        with open(xml_path, "r") as f:
            xml_data = f.read()
        bs_data = BeautifulSoup(xml_data, "xml")
        text_catma_id_to_catma_label_id = create_text_catma_id_to_label_mapping(bs_data)
        catma_label_id_to_label = create_label_catma_id_to_label_mapping(bs_data)
        catma_annotations = create_catma_annotations(bs_data)
        not_tagged_start_char_end_char = get_not_tagged_start_chars_end_chars(bs_data)

        catma_ids0 = list(set(text_catma_id_to_catma_label_id.keys()))
        catma_ids1 = list(set([catma_annotation.segment_id for catma_annotation in catma_annotations]))
        in0not1 = [katma_id for katma_id in catma_ids0 if katma_id not in catma_ids1]
        in1not0 = [katma_id for katma_id in catma_ids1 if katma_id not in catma_ids0]
        assert len(in0not1) == 0
        assert len(in1not0) == 0
        with open(txt_path) as f:
            txt_data = f.read()
        txt_data = txt_data.replace("\n", "  ").replace("\t", "  ")
        df = pd.DataFrame(text_catma_id_to_catma_label_id.items(), columns=["text_segment_catma_id", "label_catma_id"])
        df["label"] = df.apply(lambda row: catma_label_id_to_label[row["label_catma_id"]], axis=1)

        catma_annotation_df = catma_annotations_to_df(catma_annotations)
        df = pd.merge(df, catma_annotation_df, on=['text_segment_catma_id'])
        # df = df.apply(add_start_char_end_char,
        #               text_catma_id_to_start_char_end_char=catma_annotations, axis=1)
        # todo: split to sentences
        not_tagged_df = pd.DataFrame(not_tagged_start_char_end_char, columns=["start_char", "end_char"])
        not_tagged_df["text_segment_catma_id"] = None
        not_tagged_df["label_catma_id"] = None
        not_tagged_df["label"] = None
        df = pd.concat([df, not_tagged_df], ignore_index=True)
        df["file"] = os.path.basename(protocol_dir)
        committee = extract_committee_from_text(txt_data)
        df["committee"] = committee
        try:
            protocol_number = extract_protocol_number_from_text(txt_data)
        except:
            protocol_number = extract_protocol_number_from_dir(protocol_dir)
        df["protocol_number"] = protocol_number
        df["text"] = df.apply(lambda row: " ".join(txt_data[row["start_char"]:row["end_char"]].split()), axis=1)
        df["before_text_context"] = df.apply(lambda row: txt_data[row["start_char"] + BEFORE_TEXT_CONTEXT_SIZE:row[
                                                                                                                   "end_char"] + BEFORE_TEXT_CONTEXT_SIZE],
                                             axis=1)
        df["after_text_context"] = df.apply(
            lambda row: txt_data[row["start_char"] + AFTER_TEXT_CONTEXT_SIZE:row["end_char"] + AFTER_TEXT_CONTEXT_SIZE],
            axis=1)
        df = df.sort_values(by="start_char")
        dataframes.append(df)
    concatanated_df = pd.concat(dataframes, ignore_index=True)
    concatanated_df = concatanated_df.drop_duplicates(ignore_index=True)
    return concatanated_df


def create_text_catma_id_to_label_mapping(bs_data):
    text_segment_catma_id_to_label_id = dict()
    bs_text_data = bs_data.find_all("text")
    assert len(bs_text_data) == 1
    bs_text_data = bs_text_data[0]
    fss = bs_text_data.find_all("fs")
    for fs in fss:
        fs_label = fs.get("type")
        fs_text_segment_catma_id = fs.get("xml:id")
        if fs_text_segment_catma_id in text_segment_catma_id_to_label_id:
            raise NotImplemented
        text_segment_catma_id_to_label_id[fs_text_segment_catma_id] = fs_label
    return text_segment_catma_id_to_label_id


def create_label_catma_id_to_label_mapping(bs_data):
    label_catma_id_to_label = dict()
    bs_data_labels = bs_data.find_all("encodingDesc")
    assert len(bs_data_labels) == 1
    bs_data_labels = bs_data_labels[0]
    bs_data_labels_fs_decl = bs_data_labels.find_all("fsDecl")
    for bs_data_label_fs_decl in bs_data_labels_fs_decl:
        fs_descr = bs_data_label_fs_decl.find_all("fsDescr")
        assert len(fs_descr) == 1
        fs_descr = fs_descr[0]
        label = fs_descr.text
        label_catma_id = bs_data_label_fs_decl.get("type")
        if label_catma_id in label_catma_id_to_label:
            raise NotImplemented
        label_catma_id_to_label[label_catma_id] = label
    return label_catma_id_to_label


def create_catma_annotations(bs_data):
    catma_annotations = []
    bs_text_data = bs_data.find_all("text")
    assert len(bs_text_data) == 1
    bs_text_data = bs_text_data[0]
    text_segments = bs_text_data.find_all("seg")
    for text_segment in text_segments:
        catma_text_segment_ids = text_segment.get("ana").replace("#", "").split()
        ptrs = text_segment.find_all("ptr")
        assert len(ptrs) == 1
        ptrs = ptrs[0]
        match = START_CHAR_END_CHAR_EXTRACTION_REGEX.match(ptrs.get("target"))
        if not match:
            raise NotImplemented
        start_char, end_char = map(int, match.groups())
        for catma_text_segment_id in catma_text_segment_ids:
            catma_annotation = CatmaAnnotation(catma_text_segment_id, start_char, end_char)
            catma_annotations.append(catma_annotation)
            # if catma_text_segment_id in [catma_annotation.segment_id for catma_annotation in catma_annotations]: # todo: we used to elongate two following segments here
            #     old_start_char, old_end_char = text_segment_catma_id_to_start_chars_end_chars[catma_text_segment_id]
            #     if start_char != old_end_char:
            #         raise NotImplemented
            #     text_segment_catma_id_to_start_chars_end_chars[catma_text_segment_id].append((old_start_char, end_char))
            # catma_annotation = CatmaAnnotation(catma_text_segment_id, start_char, end_char)
            # catma_annotations.append(catma_annotation)
            # else:
            # text_segment_catma_id_to_start_chars_end_chars[catma_text_segment_id] = [(start_char, end_char)]
    return catma_annotations


# def add_start_char_end_char(row, text_catma_id_to_start_char_end_char):
#     start_char, end_char = text_catma_id_to_start_char_end_char[row["text_segment_catma_id"]]
#     row["start_char"] = start_char
#     row["end_char"] = end_char
#     return row


def extract_committee_from_text(text):
    match = COMMITTEE_REGEX.search(text)
    if not match:
        raise NotImplemented
    return match.group().strip()


def extract_protocol_number_from_text(text: str):
    match = PROTOCOL_NUMBER_REGEX_FROM_TEXT.search(text)
    if not match:
        raise NotImplemented
    return match.group(1).strip()


def extract_protocol_number_from_dir(dir_path: str):
    protocol_dir_name = os.path.basename(dir_path)
    match = PROTOCOL_NUMBER_REGEX_FROM_DIR_NAME.search(protocol_dir_name)
    if not match:
        raise NotImplemented
    return match.group(1).strip()


def get_not_tagged_start_chars_end_chars(bs_text_data):
    not_tagged_start_char_end_char = []
    pointer_to_tagged_or_not_segments = bs_text_data.find_all("ptr")
    for pointer_to_tagged_or_not_segment in pointer_to_tagged_or_not_segments:
        if type(pointer_to_tagged_or_not_segment.parent.get("ana")) == str:
            continue  # it is annotated and shouldn't be none
        else:
            match = START_CHAR_END_CHAR_EXTRACTION_REGEX.match(pointer_to_tagged_or_not_segment.get("target"))
            if not match:
                raise NotImplemented
            start_char, end_char = map(int, match.groups())
            not_tagged_start_char_end_char.append([start_char, end_char])
    return not_tagged_start_char_end_char
