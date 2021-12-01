# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Util functions for loading datasets.
"""
import os
import json
import numpy as np
import torch
from transformers import BertTokenizer

from lmgvp.utils import prep_seq
from lmgvp.datasets import (
    SequenceDatasetWithTarget,
    ProteinGraphDatasetWithTarget,
    BertProteinGraphDatasetWithTarget,
)
from lmgvp.deepfrier_utils import load_GO_annot

# DATA_ROOT_DIR = "/home/ec2-user/SageMaker/efs"
DATA_ROOT_DIR = "./dataset/"


def load_gvp_data(
    gvp_data_dir=DATA_ROOT_DIR,
    task="protease/with_tags",
    split="train",
    seq_only=False,
):
    """For GVP models only.
    These prepared graph data files are generated by `generate_gvp_dataset.py`

    Args:
        task: choose from ['protease/with_tags', 'Fluorescence', 'DeepFRI_GO']
        seq_only: retain only the sequences in the returned list of objects
        split: String. Split of the dataset to be loaded. One of ['train', 'valid', 'test'].
        seq_only: Bool. Wheather or not to return only sequences without coordinates.

    Retrun:
        Dictionary containing the GVP dataset of proteins.
    """
    filename = os.path.join(gvp_data_dir, task, f"proteins_{split}.json")
    dataset = json.load(open(filename, "rb"))
    if seq_only:
        # delete the "coords" in data objects
        for obj in dataset:
            obj.pop("coords", None)
    return dataset


def preprocess_seqs(tokenizer, dataset):
    """Preprocess seq in dataset and bind the input_ids, attention_mask.

    Args:
        tokenizer: hugging face artifact. Tokenization to be used in the sequence.
        dataset: Dictionary containing the GVP dataset of proteins.

    Return:
        Input dataset with `input_ids` and `attention_mask`
    """
    seqs = [prep_seq(rec["seq"]) for rec in dataset]
    encodings = tokenizer(seqs, return_tensors="pt", padding=True)
    # add input_ids, attention_mask to the json records
    for i, rec in enumerate(dataset):
        rec["input_ids"] = encodings["input_ids"][i]
        rec["attention_mask"] = encodings["attention_mask"][i]
    return dataset


def load_GO_labels(task="cc"):
    """Load the labels in the GO dataset

    Args:
        task: String. GO task. One of: 'cc', 'bp', 'mf'

    Return:
        Tuple where the first element is a dictionary mapping proteins to their target, second element is an integer with the number of outputs of the task and the third element is a matrix with the weight of each target.
    """
    prot2annot, goterms, gonames, counts = load_GO_annot(
        os.path.join(
            DATA_ROOT_DIR,
            "GeneOntology/nrPDB-GO_annot.tsv",
        )
    )
    goterms = goterms[task]
    gonames = gonames[task]
    num_outputs = len(goterms)

    # computing weights for imbalanced go classes
    class_sizes = counts[task]
    mean_class_size = np.mean(class_sizes)
    pos_weights = mean_class_size / class_sizes
    pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))
    # to tensor
    pos_weights = torch.from_numpy(pos_weights.astype(np.float32))
    return prot2annot, num_outputs, pos_weights


def add_GO_labels(dataset, prot2annot, go_ont="cc"):
    """
    Add GO labels to a dataset

    Args:
        dataset: list of dict (output from `load_gvp_data`)
        prot2annot: output from `load_GO_labels`
        go_ont: String. GO ontology/task to be used. One of: 'cc', 'bp', 'mf'

    Return:
        Dataset formatted as a list. Where, for each element (dictionary), a `target` field has been added.

    """
    for rec in dataset:
        rec["target"] = torch.from_numpy(
            prot2annot[rec["name"]][go_ont].astype(np.float32)
        )
    return dataset


def get_dataset(task="", model_type="", split="train"):
    """Load data from files, then transform into appropriate
    Dataset objects.
    Args:
        task: one of ['cc', 'bp', 'mf', 'protease', 'flu']
        model_type: one of ['seq', 'struct', 'seq_struct']
        split: one of ['train', 'valid', 'test']

    Return:
        Torch dataset.
    """
    seq_only = True if model_type == "seq" else False

    tokenizer = None
    if model_type != "struct":
        # need to add BERT
        print("Loading BertTokenizer...")
        tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )

    # Load data from files
    if task in ("cc", "bp", "mf"):  # GO dataset
        # load labels
        prot2annot, num_outputs, pos_weights = load_GO_labels(task)
        # load features
        dataset = load_gvp_data(
            task="GeneOntology", split=split, seq_only=seq_only
        )
        add_GO_labels(dataset, prot2annot, go_ont=task)
    else:
        data_dir = {"protease": "protease/with_tags", "flu": "Fluorescence"}
        dataset = load_gvp_data(
            task=data_dir[task], split=split, seq_only=seq_only
        )
        num_outputs = 1
        pos_weights = None

    # Convert data into Dataset objects
    if model_type == "seq":
        if num_outputs == 1:
            targets = torch.tensor(
                [obj["target"] for obj in dataset], dtype=torch.float32
            ).unsqueeze(-1)
        else:
            targets = [obj["target"] for obj in dataset]
        dataset = SequenceDatasetWithTarget(
            [obj["seq"] for obj in dataset],
            targets,
            tokenizer=tokenizer,
            preprocess=True,
        )
    else:
        if num_outputs == 1:
            # convert target to f32 [1] tensor
            for obj in dataset:
                obj["target"] = torch.tensor(
                    obj["target"], dtype=torch.float32
                ).unsqueeze(-1)
        if model_type == "struct":
            dataset = ProteinGraphDatasetWithTarget(dataset, preprocess=False)
        elif model_type == "seq_struct":
            dataset = preprocess_seqs(tokenizer, dataset)
            dataset = BertProteinGraphDatasetWithTarget(
                dataset, preprocess=False
            )

    dataset.num_outputs = num_outputs
    dataset.pos_weights = pos_weights
    return dataset
