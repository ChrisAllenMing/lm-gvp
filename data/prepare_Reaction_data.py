import json
import os
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import h5py
import pdb

from tqdm import tqdm
from joblib import Parallel, delayed
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

from contact_map_utils import parse_pdb_structure

# raw_path = './fold_classification/'
raw_path = './dataset/Reaction/'

# Get the label mapping
class_map_name = os.path.join(raw_path, "chain_functions.txt")

label_dict = {}
with open(class_map_name, "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line == "":
            continue

        k, v = line.split(',')
        label_dict[k] = int(v)

# Store the protein structures of each split into a json file
AA_dict = {'ALA': 'A', 'ASX': 'B', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'SEC': 'U',
'VAL': 'V', 'TRP': 'W', 'XAA': 'X', 'TYR': 'Y', 'GLX': 'Z'}


def get_seq(residue_names, residue_ids):
    curr_id = -1
    seq_list = []
    for residue_name, residue_id in zip(residue_names, residue_ids):
        if residue_id > curr_id:
            try:
                seq_list.append(three_to_one(residue_name.decode('ascii')))
            except:
                seq_list.append('X')
            curr_id = residue_id
    seq = "".join(seq_list)

    return seq


def get_coords(residue_ids, atom_names, atom_coords, target_atoms=["N", "CA", "C", "O"]):
    curr_id = -1
    coords_dict = {}
    for residue_id, atom_name, atom_coord in zip(residue_ids, atom_names, atom_coords):
        if residue_id > curr_id:
            curr_id = residue_id
            coords_dict[curr_id] = {}
        coords_dict[curr_id][atom_name.decode('ascii')] = atom_coord

    all_coords = np.zeros((len(coords_dict), len(target_atoms), 3))
    res_indices = list(coords_dict.keys())
    res_indices.sort()
    for i, res_index in enumerate(res_indices):
        res_coords_dict = coords_dict[res_index]
        for atom_idx, tgt_atom in enumerate(target_atoms):
            try:
                tgt_atom_coord = res_coords_dict[tgt_atom]
            except:
                tgt_atom_coord = np.asarray([np.nan] * 3)
            all_coords[i, atom_idx] = tgt_atom_coord

    return all_coords


splits = ['train', 'valid', 'test']
for split in splits:
    data_list = []
    split_filename = os.path.join(raw_path, "{}.txt".format(split))
    with open(split_filename, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            protein_name = line

            hdf5_filename = os.path.join(raw_path, "data", "{}.hdf5".format(protein_name))
            h5File = h5py.File(hdf5_filename, "r")
            atom_residue_names = h5File["atom_residue_names"][()]
            atom_residue_id = h5File["atom_residue_id"][()]
            atom_coords = h5File["atom_pos"][(0)]
            atom_names = h5File["atom_names"][()]

            seq = get_seq(atom_residue_names, atom_residue_id)
            coords = get_coords(atom_residue_id, atom_names, atom_coords)
            try:
                label = label_dict[protein_name]
            except:
                continue

            output = {}
            output["seq"] = seq
            output["coords"] = coords.tolist()
            output["name"] = protein_name
            output["target"] = label
            data_list.append(output)

    print("Number of {} samples: ".format(split), len(data_list))
    outfile = os.path.join(raw_path, "proteins_{}.json".format(split))
    json.dump(data_list, open(outfile, "w"))
    print("Save to: ", outfile)