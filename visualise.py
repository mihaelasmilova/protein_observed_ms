from __future__ import print_function, division
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from PIL import Image
from cairosvg import svg2png

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
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


def moltosvg(mol,molSize=(450,350),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:','')


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


def show_single_timepoint_peaks(peak_data, x_data, scaled_y_data, relative_labelling, save_direc, injection_name, timepoint, allowed_ppm_error=50.0):
    def assign_colour(ppm_error):
        if ppm_error <= allowed_ppm_error:
            colour = 'green'
        elif ppm_error <= 1.5*allowed_ppm_error:
            colour = 'orange'
        else:
            colour = 'red'
        return colour

    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()

    peak_props = dict(boxstyle='round', alpha=0.3)

    for i, p in enumerate(peak_data):
        peak_props['facecolor'] = assign_colour(p[2])
        ax.annotate(s="{}, labelling: {} %".format(str(round(p[1], 3)), round(relative_labelling[i], 2)),
                    xy=(x_data[p[0]], scaled_y_data[p[0]]),
                    fontsize=12,
                    bbox=peak_props,
                    rotation=25)
    ax.plot(x_data, scaled_y_data)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Da', fontsize=14)
    ax.set_title("{}, time: {}".format(injection_name, timepoint), fontsize=14.0)
    fig.savefig(os.path.join(save_direc, '{}_{}.png'.format(injection_name, timepoint)))
    plt.close()


def plot_compound_summary_new(compound_data,
                              injection_df_fname,
                              expected_protein_mass,
                              compound_smiles,
                              corrected_compound_mass,
                              save_dir):

    injection_df = pd.read_csv(injection_df_fname)
    injection_name = injection_df['Injection name'][0]
    image_save_fname = os.path.join(save_dir, "{}.png".format(injection_name))

    compound_mol, _, _ = analyse.get_molecule_data_from_smiles(compound_smiles)
    # Create the molecule picture:
    try:
        mol_im = moltosvg(compound_mol, molSize=(450, 350))
        svg2png(bytestring=mol_im, write_to=image_save_fname)
    except:
        pass


    timepoints = sorted(list(set(injection_df['Injection time'])))

    # Get the information about the peaks
    table_columns = ["Timepoint (min)", "Labelling %", "Double labelling %"]
    table_data = []

    expected_single_labelling = expected_protein_mass + corrected_compound_mass
    expected_double_labelling = expected_protein_mass + 2 * corrected_compound_mass

    # Plotting offsets calculation:
    x_axis_stagger = 100.0
    y_axis_stagger = 5000.0

    fig, ax1 = plt.subplots(figsize=(15, 6))
    for d in range(len(timepoints)):
        x_data = compound_data[d][0]
        y_data = compound_data[d][1]
        peaks = injection_df[injection_df['Injection time'] == timepoints[d]]
        peak_masses = peaks['Peak mass']
        # If the masses were corrected for discretisation, get them back to int
        peak_masses = np.array([int(round(m)) for m in peak_masses])
        peak_masses_idxs = [np.where(x_data == pm)[0][0] for pm in peak_masses]

        single_labelled_peak = peaks[(peaks['Ligand count'] == 1) & (peaks['Relative labelling'] > 0)]

        double_labelled_peak = peaks[(peaks['Ligand count'] == 2) & (peaks['Relative labelling'] > 0)]

        if single_labelled_peak.empty:
            table_data_1 = 0.0
        else:
            table_data_1 = np.round(single_labelled_peak['Relative labelling'].values[0], 2)

        if double_labelled_peak.empty:
            table_data_2 = 0.0
        else:
            table_data_2 = np.round(double_labelled_peak['Relative labelling'].values[0], 2)

        table_data.append([timepoints[d], table_data_1, table_data_2])

        # offset x and y data:
        x_data = x_data + d * x_axis_stagger
        y_data = y_data + d * y_axis_stagger
        peak_props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        if d == 0:
            x_add, y_add = move_labels(x_data, y_data, peak_masses_idxs)
            for i, p in enumerate(peak_masses_idxs):
                ax1.annotate(s=str(np.round(peaks['Peak mass'][i], 2)),
                             xy=(x_data[p], y_data[p]),
                             bbox=peak_props,
                             rotation=25)
        # Plot the spectra
        ax1.plot(x_data, y_data, label=timepoints[d])
        ax1.set_xticks(np.arange(x_data[0], x_data[-1], step=500))
        ax1.get_yaxis().set_visible(False)
        ax1.set_xlabel('Da', fontsize=14)
        ax1.legend(labels=['{} mins'.format(t) for t in timepoints])

    textstr = """ Unlabelled protein mass: {} Da \n Single labelled protein mass: {} Da \n Double labelled protein mass: {} Da""".format(round(expected_protein_mass,2),
                                                                                                                                        round(expected_single_labelling,2),
                                                                                                                                        round(expected_double_labelling,2))

    if compound_mol:
        png_in = image_save_fname
        im = Image.open(png_in)
        im = np.array(im)
        fig.figimage(im, fig.bbox.xmax*0.65, fig.bbox.ymax*0.40)
    else:
        pass

    plt.subplots_adjust(left=0.02, right=0.6, top=0.9, bottom=0.1)
    the_table = plt.table(cellText=table_data,
                          colLabels=table_columns,
                          loc='bottom',
                          fontsize=14.0,
                          bbox=[1.05, 0.0, 0.5, 0.3])

    ppeak_props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.6, 0.6, textstr, fontsize=10, transform=ax1.transAxes, bbox=ppeak_props)
    plt.suptitle(injection_name)
    fig.savefig(image_save_fname)
    plt.close()
    return






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

    fig, ax1 = plt.subplots(figsize=(15, 6))
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
    #ax1.text(0.6, 0.95, textstr, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    prot_props = dict(boxstyle='round', facecolor='red', alpha=0.3)
    prot_textstr = "Protein MW: {} Da \nExpected labelling: {} Da".format(expected_protein_mass, round(expected_protein_mass + corrected_compound_mw,1))
    #ax1.text(0.6, 0.15, prot_textstr, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=prot_props)

    # ax2.axis('off')
    # ax2.set_title(injection_name, fontsize=14)

    # Place a text box in lower right corner with protein data
    try:
        png_in = 'output.png'
        im = Image.open(png_in)
        im = np.array(im)
        fig.figimage(im, fig.bbox.xmax*0.65, fig.bbox.ymax*0.40)
    except TypeError:
        pass

    columns = ["Timepoint", "Peak mass", "Labelling", "Error"]
    cell_text = [['3', '1000', 'Yes', '50'], ['3', '1000', 'Yes', '50'], ['3', '1000', 'Yes', '50']]


    plt.subplots_adjust(left=0.02, right=0.6, top=0.9, bottom=0.1)
    the_table = plt.table(cellText=cell_text,
                          colLabels=columns,
                          loc='bottom',
                          bbox=[1.05, 0.0, 0.5, 0.3])

    fig.savefig(os.path.join(save_dir, "{}.png".format(injection_name)))
    plt.close()
    return

if __name__ == "__main__":
    import pandas as pd
    import csv

    timepoints = ['3 min', '1 hr', '3 hr']
    expected_protein_mass = 33796.0
    mass_window = [33500, 35000]
    split_save_dir = "../all_timepoints_injections_split"
    compounds_file = os.path.join("../Protein observe MS_raw", "200527 Shipment 4 and 5 compound list.csv")
    compounds_data = pd.read_csv(compounds_file)
    injection_names = ["{}_{}".format(j, i) for i, j in
                       zip(compounds_data['Compounds List'], compounds_data['Injections'])][:2]

    image_save_dir = "../all_analysis_images"
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









