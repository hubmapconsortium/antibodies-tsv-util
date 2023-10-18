
import logging
import re
from os import walk
from pathlib import Path
from typing import List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s - %(message)s")
logger = logging.getLogger(__name__)


SEGMENTATION_CHANNEL_NAMES = [
    "cells",
    "nuclei",
    "cell_boundaries",
    "nucleus_boundaries",
]

structured_annotation_template="""<StructuredAnnotations>
<XMLAnnotation ID="Annotation:1">
    <Value>
        <OriginalMetadata>
            <Key>ProteinIDMap</Key>
            <Value>
                {protein_id_map_sa}
            </Value>
        </OriginalMetadata>
    </Value>
</XMLAnnotation>
</StructuredAnnotations>"""


def add_structured_annotations(xml_string: str, sa_str: str) -> str:
    sa_placement = xml_string.find("</Image>") + len("</Image>")
    xml_string = xml_string[:sa_placement] + sa_str + xml_string[sa_placement:]
    return xml_string


def create_map_lines(df):
    map_lines = []
    for i in df.index:
        ch_id = df.at[i, "channel_id"]
        ch_name = get_analyte_name(df.at[i, "antibody_name"])
        uniprot_id = df.at[i, "uniprot_accession_number"]
        rr_id = df.at[i, "rr_id"]
        line = f'<Channel ID="{ch_id}" Name="{ch_name}" UniprotID="{uniprot_id}" RRID="{rr_id}"/>'
        map_lines.append(line)
    return "\n".join(map_lines)

TIFF_FILE_NAMING_PATTERN = re.compile(r"^R\d{3}_X(\d{3})_Y(\d{3})\.tif")


def generate_sa_ch_info(ch_name: str, antb_info: Optional[pd.DataFrame]) -> str:
    empty_ch_info = f'<Channel ID="None" Name="{ch_name}" UniprotID="None" RRID="None"/>'
    if antb_info is None:
        ch_info = empty_ch_info
    else:
        if ch_name in antb_info["target"].to_list():
            ch_ind = antb_info[antb_info["target"] == ch_name].index[0]
            ch_id = antb_info.at[ch_ind, "channel_id"]
            uniprot_id = antb_info.at[ch_ind, "uniprot_accession_number"]
            rr_id = antb_info.at[ch_ind, "rr_id"]
            ch_info = f'<Channel ID="{ch_id}" Name="{ch_name}" UniprotID="{uniprot_id}" RRID="{rr_id}"/>'
        else:
            ch_info = empty_ch_info
    return ch_info


def generate_structured_annotations(ch_names: List[str], antb_info: Optional[pd.DataFrame]) -> str:
    ch_infos = []
    for ch_name in ch_names:
        ch_info = generate_sa_ch_info(ch_name, antb_info)
        ch_infos.append(ch_info)
    ch_sa = "\n".join(ch_infos)
    sa = structured_annotation_template.format(protein_id_map_sa=ch_sa)
    return sa


def get_analyte_name(antibody_name: str) -> str:
    antb = re.sub(r"Anti-", "", antibody_name)
    antb = re.sub(r"\s+antibody", "", antb)
    return antb


def find_antibodies_meta(input_dir: Path) -> Optional[Path]:
    """
    Finds and returns the first metadata file for a HuBMAP data set.
    Does not check whether the dataset ID (32 hex characters) matches
    the directory name, nor whether there might be multiple metadata files.
    """
    #possible_dirs = [input_dir, input_dir / "extras"]
    metadata_filename_pattern = re.compile(r"^[0-9A-Za-z\-_]*antibodies\.tsv$")
    found_files = []
    for dirpath, dirnames, filenames in walk(input_dir):
        for filename in filenames:
            if metadata_filename_pattern.match(filename):
                found_files.append(Path(dirpath) / filename)

    if len(found_files) == 0:
        logger.warning("No antibody.tsv file found")
        antb_path = None
    else:
        antb_path = found_files[0]
    return antb_path


def sort_by_cycle(antb_path: Path):
    df = pd.read_table(antb_path)
    cycle_channel_pattern = re.compile(r'cycle(?P<cycle>\d+)_ch(?P<channel>\d+)', re.IGNORECASE)
    searches = [cycle_channel_pattern.search(v) for v in df['channel_id']]
    cycles = [int(s.group('cycle')) for s in searches]
    channels = [int(s.group('channel')) for s in searches]
    df.index = [cycles, channels]
    df = df.sort_index()
    return df


def get_ch_info_from_antibodies_meta(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    #df = df.set_index("channel_id", inplace=False)
    antb_names = df["antibody_name"].to_list()
    antb_targets = [get_analyte_name(antb) for antb in antb_names]
    df["target"] = antb_targets
    return df


def replace_provider_ch_names_with_antb(provider_ch_names: List[str], antb_ch_info: pd.DataFrame):
    targets = antb_ch_info["target"].to_list()
    corrected_ch_names = []
    for ch in provider_ch_names:
        new_ch_name = ch
        for t in targets:
            if re.match(ch, t, re.IGNORECASE):
                new_ch_name = t
                break
        corrected_ch_names.append(new_ch_name)
    return corrected_ch_names
