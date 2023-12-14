import logging
import re
from os import walk
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tifffile import TiffFile
from ome_types.model import MapAnnotation, StructuredAnnotationList, Map

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s - %(message)s")
logger = logging.getLogger(__name__)

metadata_filename_pattern = re.compile(r"^[0-9A-Fa-f]{32}antibodies\.tsv$")


def add_cycle_channel_numbers(channel_names: List[str]) -> List[str]:
    """
    Adds cycle and channel info during the collect dataset info step. Replaces a similar function that adds a number on the end of duplicate channel names.
    """
    new_names = []
    cycle_count = 1
    channel_count = 1

    for original_name in channel_names:
        new_name = f"cyc{cycle_count}_ch{channel_count}_orig{original_name}"
        new_names.append(new_name)

        channel_count += 1
        if channel_count > 4:  # Assuming 4 channels per cycle, modify accordingly
            channel_count = 1
            cycle_count += 1

    return new_names


def generate_sa_ch_info(
    channel_id: str,
    og_ch_names_info: pd.Series,
    antb_info: Optional[pd.DataFrame],
) -> Optional[MapAnnotation]:
    if antb_info is None:
        return None
    cycle, channel = og_ch_names_info["Cycle"], og_ch_names_info["Channel"]
    try:
        antb_row = antb_info.loc[(cycle, channel), :]
    except KeyError:
        return None

    uniprot_id = antb_row["uniprot_accession_number"]
    rrid = antb_row["rr_id"]
    antb_id = antb_row["channel_id"]
    ch_key = Map.M(k="Channel ID", value=channel_id)
    name_key = Map.M(k="Name", value=antb_row["target"])
    og_name_key = Map.M(k="Original Name", value=og_ch_names_info["channel_name"])
    uniprot_key = Map.M(k="UniprotID", value=uniprot_id)
    rrid_key = Map.M(k="RRID", value=rrid)
    antb_id_key = Map.M(k="AntibodiesTsvID", value=antb_id)
    ch_info = Map(ms=[ch_key, name_key, og_name_key, uniprot_key, rrid_key, antb_id_key])
    annotation = MapAnnotation(value=ch_info)
    return annotation


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
    og_ch_names_df["Cycle"] = pd.to_numeric(og_ch_names_df["Cycle"])
    og_ch_names_df["Channel"] = pd.to_numeric(og_ch_names_df["Channel"])
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


def get_original_names(og_ch_names_df: pd.DataFrame, antibodies_df: pd.DataFrame) -> List[str]:
    mapping = map_cycles_and_channels(antibodies_df)
    original_channel_names = []
    for i in og_ch_names_df.index:
        channel_id = og_ch_names_df.at[i, "channel_id"].lower()
        original_name = og_ch_names_df.at[i, "channel_name"]
        target = mapping.get(channel_id, None)
        if target is not None:
            original_channel_names.append(original_name)
        else:
            original_channel_names.append("None")
    return original_channel_names


def generate_structured_annotations(xml, channel_ids, og_ch_names_df, antb_info, channelNames ):
    original_channel_names = get_original_names(og_ch_names_df, antb_info)
    annotations = StructuredAnnotationList()
    for i in range(len(channelNames)):
        xml.images[0].pixels.channels[i].name = channelNames[i]
        channel_id = channel_ids[i]
        xml.images[0].pixels.channels[i].id = channel_id
        # Extract channel information for structured annotations
        ch_name = channelNames[i]
        original_name = original_channel_names[i]
        ch_info = generate_sa_ch_info(ch_name, original_name, antb_info, channel_id)
        annotations.append(ch_info)
    xml.structured_annotations = annotations
    return xml
