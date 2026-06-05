import yaml
import numpy as np
import pandas as pd
import MDAnalysis as mda
from typing import Dict
from pathlib import Path
from .utils import dataframe_to_mda_universe

embedding_map_fivebead = {
    "ALA": 1,
    "CYS": 2,
    "ASP": 3,
    "GLU": 4,
    "PHE": 5,
    "GLY": 6,
    "HIS": 7,
    "ILE": 8,
    "LYS": 9,
    "LEU": 10,
    "NLE": 10,  # Type Norleucine as Leucine
    "MET": 11,
    "ASN": 12,
    "PRO": 13,
    "GLN": 14,
    "ARG": 15,
    "SER": 16,
    "THR": 17,
    "VAL": 18,
    "TRP": 19,
    "TYR": 20,
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
}


class CGEmbeddingMap(dict):
    """
    General class for defining embedding maps as Dict
    """

    def __init__(self, embedding_map_dict: Dict[str, int]):
        for k, v in embedding_map_dict.items():
            self[k] = v


class CGEmbeddingMapFiveBead(CGEmbeddingMap):
    """
    Five-bead embedding map defined by:
        - N : backbone nitrogen
        - CA : backbone alpha carbon (specialized for glycing)
        - C : backbone carbonyl carbon
        - O : backbone carbonyl oxygen
        - CB : residue-specific beta carbon
    """

    def __init__(self):
        super().__init__(embedding_map_fivebead)


class CGEmbeddingMapCA(CGEmbeddingMap):
    """
    One-bead embedding map defined by:
        - CA : backbone alpha carbon, carrying aminoacid identity
    """

    def __init__(self):
        ca_dict = {key: emb for key, emb in embedding_map_fivebead.items() if emb <= 20}
        super().__init__(ca_dict)

# load mappings at import time
_VIRTUAL_MAPPINGS = {}
for fname in Path('mappings').glob("*.yml"):
    source, destination = fname.stem.split("_to_")
    with open(fname) as file:
        if source not in _VIRTUAL_MAPPINGS:
            _VIRTUAL_MAPPINGS[source] = {}
        _VIRTUAL_MAPPINGS[source][destination] = yaml.safe_load(file)

_TERMINAL_BEAD_TYPE = 'P6'


class CGEmbeddingMapMartini(CGEmbeddingMap):
    """Martini virtual beads mapping.
    Initializes a dict from bead types to index."""

    def __init__(self, destination='martini3'):
        bead_types = {}
        bead_types[_TERMINAL_BEAD_TYPE] = len(bead_types)  # martini capped
        for residue_mapping in _VIRTUAL_MAPPINGS['charmm36'][destination].values():
            for bead_name in list(residue_mapping)[:-2]:
                bead_type = residue_mapping[bead_name]['type']
                if bead_type not in bead_types:
                    bead_types[bead_type] = len(bead_types)
        super().__init__(bead_types)
            

all_residues = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def embedding_fivebead(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_fivebead[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_fivebead["GLY"]
        else:
            atom_type = embedding_map_fivebead[name]
    elif name == "CB":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_ca(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_martini(atom_df,
                      source='charmm36',
                      destination='martini3',
                      select=('not water and '
                              'not resname SOD CLA POT MAG CAL ION'),
                      terminal_beads=['BB']):
    """
    Function mapping high-resolution topology to Martini embedding map.
    It automatically excludes solvent molecules.
    
    Parameters
    ----------
    atom_df : pd.DataFrame
        DataFrame containing atom information.
    source : str
        Source forcefield of the system, used to determine the correct
        mapping. Currently only "charmm36" is supported, but more can be
        added by including the corresponding mapping in the 'mappings'
        directory.
    destination : str
        Destination forcefield of the system, used to determine the correct
        mapping. Currently only "martini3" is supported, but more can be
        added by including the corresponding mapping in the 'mappings'
        directory.
    terminal_beads : str
        Beads to replace with _TERMINAL_BEAD_TYPE if terminal residues.
    
    Returns
    -------
    cg_df : pd.DataFrame
        DataFrame containing CG information.

    Remarks
    -------
    This function returns a different object than the other embedding
    functions in input_generator.embedding_maps. It can be used as an input
    argument of MMSampleCollection.apply_cg_mapping, but *not* in
    SampleCollection.apply_cg_mapping.
    """
    if source not in _VIRTUAL_MAPPINGS:
        raise ValueError(f'Source forcefield {source!r} not available.')
    if destination not in _VIRTUAL_MAPPINGS[source]:
        raise ValueError(f'Destination forcefield {destination!r} '
                         f'not available for source {source!r}.')
    mapping = _VIRTUAL_MAPPINGS[source][destination]

    # make advantage of MDAnalysis
    universe = dataframe_to_mda_universe(atom_df)
    atoms = universe.select_atoms(select)
    to_keep = np.zeros(len(universe.atoms), dtype=bool)
    to_keep[atoms.indices] = True

    # initialize mapped beads
    bead_names = []
    bead_types = []
    bead_resids = []
    bead_resnames = []
    bead_chainID = []
    bead_aa_map = []  # atom indices
    bead_aa_weights = []  # weight associated to atom indices
    bead_bonds = []
    for chainID, chain in enumerate(atoms.segments):
        residues = chain.residues
        n_residues = len(residues)
        for chain_resindex in range(len(residues)):
            residue = residues[chain_resindex]
            if not to_keep[residue.atoms.indices[0]]:
                continue
            resname = residue.resname
            resid = int(residue.resid)
            residue_atoms = residue.atoms
            residue_atom_names = residue_atoms.names.tolist()
            is_terminal = (chain_resindex == 0) or (chain_resindex == n_residues - 1)
            residue_mapping = mapping[residue.resname]
            bead_names_in_residue = list(residue_mapping)[:-2]
            residue_bonds = residue_mapping['bonds']
            residue_offset = len(bead_names)
            for bead_name in bead_names_in_residue:
                bead_mapping = residue_mapping[bead_name]
                bead_names.append(bead_name)
                # change bead type
                if is_terminal and bead_name in terminal_beads:
                    bead_types.append(_TERMINAL_BEAD_TYPE)
                else:  # bead type defined by force field
                    bead_types.append(bead_mapping['type'])
                bead_resids.append(resid)
                bead_resnames.append(resname)
                bead_chainID.append(chainID)
                bead_aa_map.append([])
                bead_aa_weights.append([])
                bead_bonds.append([])
                # find mapping
                for atom_name, atom_weight_in_bead in \
                    bead_mapping['atoms'].items():
                    if atom_name in residue_atom_names:
                        atom_index = residue_atoms[
                            residue_atom_names.index(atom_name)].index
                        bead_aa_map[-1].append(atom_index)
                        bead_aa_weights[-1].append(atom_weight_in_bead)
            # find bonds
            for bead1_name, bead2_name in (residue_bonds or []):
                bead1_index_in_residue = bead_names_in_residue.index(bead1_name)
                bead2_index_in_residue = bead_names_in_residue.index(bead2_name)
                bead1_index = residue_offset + bead1_index_in_residue
                bead2_index = residue_offset + bead2_index_in_residue
                bead_bonds[bead1_index].append(bead2_index)
            # chain bond to the previous residue
            if residue_mapping['chain'] and chain_resindex:
                bead1_name, bead2_name = residue_mapping['chain']
                bead1_index_in_residue = bead_names_in_residue.index(bead1_name)
                bead1_index = residue_offset + bead1_index_in_residue
                bead2_index = residue_offset - 1
                while bead_names[bead2_index] != bead2_name:
                    bead2_index -= 1
                bead_bonds[bead2_index].append(bead1_index)
    
    # create dataframe
    cg_df = pd.DataFrame()
    cg_df['serial'] = np.arange(len(bead_names))
    cg_df['name'] = bead_names
    cg_df['type'] = bead_types
    cg_df['resSeq'] = bead_resids
    cg_df['resName'] = bead_resnames
    cg_df['chainID'] = bead_chainID
    cg_df['aa_map'] = bead_aa_map
    cg_df['aa_weights'] = bead_aa_weights
    cg_df['bonds'] = bead_bonds
    cg_df['element'] = 'C'  # "coarse-grained"
    return cg_df
