import glob
import os
import shutil
from typing import List
from collections import Counter
import pandas as pd

from dataset_creation.config import CATMA_DATA_DIR, AVICHAI_ZIP_FILE, OUTPUT_TSV_PATH
from dataset_creation.utils import create_dataframe


def unpack_avichai_zip_file_and_return_valid_annotation_files(catma_data_dir: str, avichai_zip_file: str) -> List[str]:
    shutil.unpack_archive(os.path.join(catma_data_dir, avichai_zip_file), catma_data_dir)
    tar_gz_paths = glob.glob(os.path.join(catma_data_dir, "*.tar.gz"))
    unpacked_archives_dir = os.path.join(catma_data_dir, "unpacked_archives")
    if not os.path.exists(unpacked_archives_dir):
        os.mkdir(unpacked_archives_dir)
    for tar_gz_path in tar_gz_paths:
        shutil.unpack_archive(tar_gz_path, unpacked_archives_dir)
    protocol_dirs = glob.glob(os.path.join(unpacked_archives_dir, "*"))
    protocol_dirs_with_annotation = []
    protocol_dirs_without_annotation = []
    for protocol_dir in protocol_dirs:
        if "annotationcollections" in os.listdir(protocol_dir):
            protocol_dirs_with_annotation.append(protocol_dir)
        else:
            protocol_dirs_without_annotation.append(protocol_dir)
    print(f"{len(protocol_dirs_with_annotation)} out of {len(protocol_dirs)} protocol dirs contain annotation")
    print(f"Protocol dirs without annotation for instance are: {protocol_dirs_without_annotation[:3]}")
    return protocol_dirs_with_annotation


if __name__ == "__main__":
    protocol_dirs_with_annotation = unpack_avichai_zip_file_and_return_valid_annotation_files(CATMA_DATA_DIR,
                                                                                              AVICHAI_ZIP_FILE)
    dfs = []
    for protocol_dir in protocol_dirs_with_annotation:
        df = create_dataframe(protocol_dir)
        dfs.append(df)
    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df = concatenated_df.drop_duplicates(ignore_index=True)
    concatenated_df["label"].replace({"judicial decision turns turns": "Judicial decision",
                                      "Anticipating Judicial Review turns": "Anticipating Judicial Review"},
                                     inplace=True)
    concatenated_df = concatenated_df[concatenated_df["label"] != "Doubt"]
    print(concatenated_df.head())
    print(concatenated_df)
    # todo: check catma ids with multiple labels
    concatenated_df.to_csv(OUTPUT_TSV_PATH, sep="\t")
    # # todo: concatenated df should be on a sentence level so we need to split by [".", ":", "?"]
