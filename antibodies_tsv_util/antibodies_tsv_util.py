import logging
import re
from os import walk
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tifffile import TiffFile

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s - %(message)s")
logger = logging.getLogger(__name__)

structured_annotation_template = """<StructuredAnnotations>
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

metadata_filename_pattern = re.compile(r"^[0-9A-Fa-f]{32}antibodies\.tsv$")


def add_structured_annotations(xml_string: str, sa_str: str) -> str:
    """
    Adds structured annotation at correct location. May need to be used in a loop depending on use.
    """
    sa_placement = xml_string.find("</Image>") + len("</Image>")
    xml_string = xml_string[:sa_placement] + sa_str + xml_string[sa_placement:]
    return xml_string


def generate_sa_ch_info(
    ch_name: str, og_name: str, antb_info: Optional[pd.DataFrame], channel_id
) -> str:
    """
    Extracts channel info from the original metadata and antibodies.tsv for the structured annotations.
    """
    empty_ch_info = f'<Channel ID="{channel_id}" Name="{ch_name}" OriginalName="None" UniprotID="None" RRID="None" AntibodiesTsvID="None"/>'
    if antb_info is None:
        ch_info = empty_ch_info
    else:
        if ch_name in antb_info["target"].to_list():
            ch_ind = antb_info[antb_info["target"] == ch_name].index[0]
            new_ch_name = antb_info.at[ch_ind, "target"]
            uniprot_id = antb_info.at[ch_ind, "uniprot_accession_number"]
            rr_id = antb_info.at[ch_ind, "rr_id"]
            antb_id = antb_info.at[ch_ind, "channel_id"]
            original_name = og_name
            ch_info = f'<Channel ID="{channel_id}" Name="{new_ch_name}" OriginalName="{original_name}" UniprotID="{uniprot_id}" RRID="{rr_id}" AntibodiesTsvID="{antb_id}"/>'
            return ch_info
        else:
            ch_info = empty_ch_info
    return ch_info


def get_analyte_name(antibody_name: str) -> str:
    """
    Strips unnecessary prefixes and suffixes off of antibody name from antibodies.tsv.
    """
    antb = re.sub(r"Anti-", "", antibody_name)
    antb = re.sub(r"\s+antibody", "", antb)
    return antb


def find_antibodies_meta(input_dir: Path) -> Optional[Path]:
    """
    Finds and returns the first metadata file for a HuBMAP data set.
    Does not check whether the dataset ID (32 hex characters) matches
    the directory name, nor whether there might be multiple metadata files.
    """
    # possible_dirs = [input_dir, input_dir / "extras"]
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
    """
    Sorts antibodies.tsv by cycle and channel number. The original tsv is not sorted correctly.
    """
    df = pd.read_table(antb_path)
    cycle_channel_pattern = re.compile(r"cycle(?P<cycle>\d+)_ch(?P<channel>\d+)", re.IGNORECASE)
    searches = [cycle_channel_pattern.search(v) for v in df["channel_id"]]
    cycles = [int(s.group("cycle")) for s in searches]
    channels = [int(s.group("channel")) for s in searches]
    df.index = [cycles, channels]
    df = df.sort_index()
    return df


def get_ch_info_from_antibodies_meta(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Adds "target" column with the antibody name that we want to replace.
    """
    # df = df.set_index("channel_id", inplace=False)
    antb_names = df["antibody_name"].to_list()
    antb_targets = [get_analyte_name(antb) for antb in antb_names]
    df["target"] = antb_targets
    return df


def map_cycles_and_channels(antibodies_df: pd.DataFrame) -> dict:
    """
    Maps cycle and channel numbers to later update channel names to the ones in antibodies.tsv.
    """
    channel_mapping = {
        channel_id.lower(): target
        for channel_id, target in zip(antibodies_df["channel_id"], antibodies_df["target"])
    }
    return channel_mapping


def replace_provider_ch_names_with_antb(
    og_ch_names_df: pd.DataFrame, antibodies_df: pd.DataFrame
) -> List[str]:
    """
    Uses cycle and channel mapping to replace the channel name with the one in antibodies.tsv.
    """
    updated_channel_names = []
    mapping = map_cycles_and_channels(antibodies_df)
    for i in og_ch_names_df.index:
        channel_id = og_ch_names_df.at[i, "channel_id"].lower()
        original_name = og_ch_names_df.at[i, "channel_name"]
        target = mapping.get(channel_id, None)
        if target is not None:
            updated_channel_names.append(target)
        else:
            updated_channel_names.append(original_name)
    return updated_channel_names


def collect_expressions_extract_channels(extractFile: Path) -> List[str]:
    """
    Given a TIFF file path, read file with TiffFile to get Labels attribute from
    ImageJ metadata. Return a list of the channel names in the same order as they
    appear in the ImageJ metadata.
    We need to do this to get the channel names in the correct order, and the
    ImageJ "Labels" attribute isn't picked up by AICSImageIO.
    """

    with TiffFile(str(extractFile.absolute())) as TF:
        ij_meta = TF.imagej_metadata
    numChannels = int(ij_meta["channels"])
    channelList = ij_meta["Labels"][0:numChannels]

    # Remove "proc_" from the start of the channel names.
    procPattern = re.compile(r"^proc_(.*)")
    channelList = [procPattern.match(channel).group(1) for channel in channelList]
    return channelList


def create_original_channel_names_df(channelList: List[str]) -> pd.DataFrame:
    """
    Creates a dataframe with the original channel names, cycle numbers, and channel numbers.
    """
    # Separate channel and cycle info from channel names and remove "orig"
    cyc_ch_pattern = re.compile(r"cyc(\d+)_ch(\d+)_orig(.*)")
    og_ch_names_df = pd.DataFrame(channelList, columns=["Original_Channel_Name"])
    og_ch_names_df[["Cycle", "Channel", "channel_name"]] = og_ch_names_df[
        "Original_Channel_Name"
    ].str.extract(cyc_ch_pattern)
    og_ch_names_df["channel_id"] = (
        "cycle" + og_ch_names_df["Cycle"] + "_ch" + og_ch_names_df["Channel"]
    )

    return og_ch_names_df


def generate_channel_ids(channelNames: List[str]) -> List[str]:
    """
    Generates a unique channel ID in the form of "Channel:O:{channel#}".
    """
    channel_ids = [f"Channel:0:{i}" for i in range(len(channelNames))]
    return channel_ids


def generate_structured_annotations(
    xml_str: str,
    channel_ids: List[str],
    updated_channel_names: List[str],
    original_channel_names: List[str],
    antibodies_df: pd.DataFrame,
) -> str:
    """
    Generates each structured annotation and returns a string off all of the structured annotations.
    """
    full_ch_info = ""
    for i in range(0, len(updated_channel_names)):
        xml_str.image().Pixels.Channel(i).Name = updated_channel_names[i]
        channel_id = channel_ids[i]
        xml_str.image().Pixels.Channel(i).ID = channel_id
        # Extract channel information for structured annotations
        ch_name = updated_channel_names[i]
        original_name = original_channel_names[i]
        ch_info = generate_sa_ch_info(ch_name, original_name, antibodies_df, channel_id)
        full_ch_info = full_ch_info + ch_info
    struct_annot = add_structured_annotations(xml_str.to_xml("utf-8"), full_ch_info)
    return struct_annot
