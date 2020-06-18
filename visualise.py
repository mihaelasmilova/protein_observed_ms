from __future__ import print_function, division
import os

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import DrawingOptions
from collections import defaultdict
import analyse


def scale_y_data(compound_data, scaling_factor):
    scaled_y_data = []
    for d in range(len(compound_data)):
        n_data = compound_data[d][:, 2]
        n_data = np.divide(n_data, np.max(n_data)) * scaling_factor
        scaled_y_data.append(n_data)
    scaled_y_data = np.array(scaled_y_data)
    return scaled_y_data


def draw_mol_rdkit(compound_mol):
    DrawingOptions.bondLineWidth = 3.5
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.atomLabelFontSize = 55
    try:
        im = Chem.Draw.MolToImage(compound_mol, size=(200, 200))
        return im
    except ValueError:
        return

def crop_to_mass_window(compound_data, mass_window):
    mass_low = mass_window[0]
    mass_high = mass_window[1]

    low_mw_index = np.where(compound_data[:, 1] == mass_low)[0][0]
    high_mw_index = np.where(compound_data[:, 1] == mass_high)[0][0]
    cropped_compound_data = compound_data[low_mw_index:high_mw_index, :]

    return cropped_compound_data

def move_labels(xdata,ydata,peaks):
    """
    Cosmetic. Used to slightly move peak labels displayed too close to each other, so that they can be read.
    The distances are hard coded, so may behave unpredictably if the figure size is changed.
    If peak labels are moving around randomly, try not calling this function.
    :param xdata:
    :param ydata:
    :param peaks:
    :return:
    """
    x_add=defaultdict(int)
    y_add=defaultdict(int)
    for i in range(len(peaks)-2):
        pk1=peaks[i]
        pk2=peaks[i+1]
        pk3=peaks[i+2]
        for p,p1 in [(pk1,pk2),(pk1,pk3)]:
            if abs(xdata[p1]-xdata[p])<90 and abs(ydata[p1]-ydata[p])<3100:
                x_add[p]-=25
                x_add[p1]+=25
                y_add[p]+=1400
                y_add[p1]-=1400

    return (x_add,y_add)


def plot_compound_summary(injection_name,
                          compound_data,
                          timepoints,
                          compound_smiles,
                          expected_protein_mass,
                          save_dir='',
                          compound_mw_correction=0.0,
                          x_axis_stagger=100,
                          y_axis_stagger=10000,
                          scaling_factor=50000):
    """
    Creates images for the compounds and compound spectra
    :param injection_name: Name of the injection + compound
    :param compound_data: list of 2d numpy arrays [timepoints: [timepoint_data[Da]: timepoint_data[counts]]
    :param timepoints: list, eg ['1 hr', '2 mins'] needed for legend
    :param compound_smiles: string,
    :param expected_protein_mass: float, mass of protein with no compound
    :param save_dir: str, path to where the image will be stored
    :param compound_mw_correction: optional, supply for covalent compounds
    :param x_axis_stagger: these control how far apart the timepoint spectra are staggered.
    :param y_axis_stagger:
    :param scaling_factor: re-scales y axis data
    :return:
    """
    # Get molecule data
    compound_mol, compound_mw, corrected_compound_mw = analyse.get_molecule_data_from_smiles(compound_smiles, compound_mw_correction)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Show the spectra
    # Scale the data. To view raw counts, can change this to scaled_y_data = compound_data, but plots may be harder to see.
    scaled_y_data = scale_y_data(compound_data, scaling_factor)

    for d in range(len(compound_data)):
        x_data = compound_data[d][:, 1] + d * x_axis_stagger
        y_data = scaled_y_data[d] + d * y_axis_stagger

        # Add the peaks to the first trace
        peak_props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        if d == 0:
            #peaks = find_peaks(y_data, height=2500, distance=40)[0]
            peaks, _ = find_peaks(y_data, prominence=1500, distance=10)
            x_add,y_add=move_labels(x_data,y_data,peaks)
            for p in peaks:
                ax1.annotate(s=str(x_data[p]),
                             xy=(x_data[p]+x_add[p], y_data[p]+y_add[p]),
                             rotation=45,
                             bbox=peak_props)
        # Plot the spectra
        ax1.plot(x_data, y_data, label=timepoints[d])
        ax1.set_xticks(np.arange(x_data[0], x_data[-1], step=500))
        ax1.get_yaxis().set_visible(False)
        ax1.set_xlabel('Da', fontsize=14)
        ax1.legend()

    # place a text box in upper right showing compound MW and expected compound MW
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'Molecular weight: {} Da \nExpected labelling: {} Da'.format(round(compound_mw, 1),
                                                                                   round(corrected_compound_mw, 1))
    ax2.text(0.6, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    prot_props = dict(boxstyle='round', facecolor='red', alpha=0.3)
    prot_textstr = "Protein MW: {} Da \nExpected labelling: {} Da".format(expected_protein_mass,
                                                                          round(expected_protein_mass + corrected_compound_mw,1))
    ax2.text(0.6, 0.15, prot_textstr, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=prot_props)

    ax2.axis('off')
    ax2.set_title(injection_name, fontsize=14)

    # Place a text box in lower right corner with protein data
    try:
        ax2.imshow(draw_mol_rdkit(compound_mol))
    except TypeError:
        pass
    fig.savefig(os.path.join(save_dir, "{}.png".format(injection_name)))
    plt.close()
    return

if __name__ == "__main__":
    import pandas as pd
    import csv

    timepoints = ['3 min', '1 hr', '3 hr']
    expected_protein_mass = 33796.0
    mass_window = [33500, 35000]
    split_save_dir = "all_timepoints_injections_split"
    compounds_file = os.path.join("Protein observe MS_raw", "200527 Shipment 4 and 5 compound list.csv")
    compounds_data = pd.read_csv(compounds_file)
    injection_names = ["{}_{}".format(j, i) for i, j in
                       zip(compounds_data['Compounds List'], compounds_data['Injections'])]

    image_save_dir = "all_analysis_images"
    if not os.path.exists(image_save_dir): os.mkdir(image_save_dir)

    image_names = ['{}.csv'.format(i) for i in injection_names]
    with open('image_names.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter='\n')
        wr.writerow(image_names)

    for i in range(len(injection_names)):
        iname = injection_names[i]
        comp_mw_correction = compounds_data['mass changes compare to MW of compounds'][i]
        comp_smiles = compounds_data['Smiles'][i]

        data1 = np.genfromtxt(os.path.join(split_save_dir, '3m', "{}.csv".format(iname)), delimiter=',')
        data2 = np.genfromtxt(os.path.join(split_save_dir, '1h', "{}.csv".format(iname)), delimiter=',')
        data3 = np.genfromtxt(os.path.join(split_save_dir, '3h', "{}.csv".format(iname)), delimiter=',')

        all_data_compound = [data1, data2, data3]

        cropped_data_compound = [crop_to_mass_window(adc, mass_window) for adc in all_data_compound]

        plot_compound_summary(injection_name=iname,
                              compound_data=cropped_data_compound,
                              timepoints=timepoints,
                              compound_smiles=comp_smiles,
                              expected_protein_mass=expected_protein_mass,
                              save_dir=image_save_dir,
                              compound_mw_correction=comp_mw_correction,
                              x_axis_stagger=100,
                              y_axis_stagger=10000,
                              scaling_factor=50000)









