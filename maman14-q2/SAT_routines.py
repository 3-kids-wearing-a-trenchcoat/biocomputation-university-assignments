from __future__ import annotations
import binding, strand, anneal, unravel, magnetic, SATinit, sequences
from SATinit import variable_rep_false, variable_rep_true, connector_rep, complement_rep
from TwoBitArray import TwoBitArray
from typing import List, Tuple
from tqdm import tqdm, trange


def check_and_reindex(threshold:float) -> None:
    """Check if there's a need to compactify the data, if so, reindex."""
    reindex_binding= binding.get_dead_fraction() > threshold
    # reindex strands (and compactify sequences) if either strands or sequences are above threshold
    reindex_strand = strand.get_dead_fraction() > threshold or sequences.get_dead_fraction() > threshold
    # if strands above threshold, reindex and get the new IDs
    new_ids = strand.reindex() if reindex_strand else None
    if reindex_binding or reindex_strand:   # if bindings are above threshold OR strands were reindexed
        binding.reindex(new_ids)            # Will also update IDs if strand was reindexed

