from __future__ import print_function, division
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import MolWt

import numpy as np
from scipy.signal import find_peaks


def get_molecule_data_from_smiles(compound_smiles, compound_mw_correction):
    try:
        compound_mol = Chem.MolFromSmiles(compound_smiles)
        AllChem.Compute2DCoords(compound_mol)
        compound_mw = MolWt(compound_mol)
        corrected_compound_mw = compound_mw + compound_mw_correction

    except TypeError:
        compound_mw = 0.0
        corrected_compound_mw = 0.0
        compound_mol = None

    return compound_mol, compound_mw, corrected_compound_mw

def protein_mw_from_prepared_pdb(prepared_protein_pdb_file, oligo_state=1):
    """
    Given the path to a protonated protein with waters, ligands, and ions removed.
    :param prepared_protein_pdb_file: str, path to the prepared pdb file
    :param oligo_state: int, oligomerisation state of the protein. default is 1 (monomer)
    :return: protein molecular weight as float
    """
    rd_pdb = Chem.MolFromPDBFile(prepared_protein_pdb_file, sanitize=False, removeHs=False)
    analytical_mass = MolWt(rd_pdb) * oligo_state
    return analytical_mass


def protein_mw_from_sequence(fasta, oligo_state=1):
    """
    Calculates the protein mass from a
    :param fasta: Single latter amino acid protein sequence 'SGFRK...'
    :param oligo_state: int, oligomerisation state of the protein. default is 1 (monomer)
    :return: protein_mw, float
    """
    analysed_seq = ProteinAnalysis(fasta)
    theory_mw_protein = analysed_seq.molecular_weight() * float(oligo_state)
    return theory_mw_protein


def mass_error_ppm(observed_mw, theory_mw):
    """

    :param observed_mw:
    :param theory_mw:
    :return: the error in ppm
    """
    return abs((observed_mw - theory_mw) / theory_mw) * 1000000


def correct_peak_mass(peak_index, x_data, y_data, p_range=5):
    """
    Corrects for the discretisation of Da in the input data by taking the weighted mean of the highest points
    in the peak (p_range number of points on either side of the peak index).
    :param peak_index:
    :param x_data:
    :param y_data:
    :param p_range:
    :return:
    """
    local_x = x_data[peak_index-p_range: peak_index+p_range]
    local_y = y_data[peak_index-p_range: peak_index+p_range]
    transposed_local_y = local_y - np.min(local_y)
    corrected_position = np.divide(np.sum(local_x*transposed_local_y), np.sum(transposed_local_y))
    return corrected_position


def assign_peaks(daltons, scaled_counts, theory_mw_protein, theory_mw_ligand, prominence, correct_peak_positions=True):
    peaks, peak_info = find_peaks(scaled_counts, prominence)
    if correct_peak_positions:
        peak_masses = [correct_peak_mass(p, daltons, scaled_counts, p_range=5) for p in peaks]
    else:
        peak_masses = daltons[peaks]

    max_peak_mass = np.max(peak_masses)
    target_mass = theory_mw_protein
    ligand_count = 0
    peak_data = []

    while target_mass <= max_peak_mass + 5.0:
        ppm_errors = np.array([mass_error_ppm(peak_mass, target_mass) for peak_mass in peak_masses])
        min_ppm_idx = np.where(ppm_errors == np.min(ppm_errors))[0][0]
        peak_observed_mass = peak_masses[min_ppm_idx]
        peak_index = peaks[min_ppm_idx]
        peak_data.append([peak_index, peak_observed_mass, target_mass, ligand_count, np.min(ppm_errors)])

        ligand_count += 1
        # Avoid getting stuck in endless loop with the negative control...
        if theory_mw_ligand >0.0:
            target_mass += ligand_count*theory_mw_ligand
        else:
            break

    return peak_data


def calculate_relative_labelling(peak_data, scaled_counts, ppm_error=5.0):
    unlabelled_protein_peak_index = peak_data[0][0]
    unlabelled_protein_peak_counts = scaled_counts[unlabelled_protein_peak_index]

    labelled_peak_idxs = []

    for p in peak_data[1:]:
        if p[-1]<=ppm_error:
            labelled_peak_idxs.append(p[0])

    total_labelled_counts = sum([scaled_counts[p_indx] for p_indx in labelled_peak_idxs])

    percent_contribs = []
    for p in peak_data:
        if p[0] in labelled_peak_idxs:
            labelled_ratio = np.divide(scaled_counts[p[0]]*100, (unlabelled_protein_peak_counts + total_labelled_counts))
            percent_contribs.append(labelled_ratio)
        else:
            percent_contribs.append(0.0)

    return percent_contribs

