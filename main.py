from __future__ import print_function, division
import os
import logging
import numpy as np
import pandas as pd
import analyse
import visualise
import parse_agilent_raw_data

def main(data_files_dict, timepoints, compound_data_path, pipeline_output_path, protein_sequence, protein_oligo_state, mass_window, peak_prominence, ppm_error=5.0, correct_peak_discretisation=True, display_mass_window=None):
    """

    :param data_files_dict:
    :param timepoints: int, the timepoint in minutes
    :param compound_data_path:
    :return:
    """
    compound_data_df = pd.read_csv(compound_data_path)
    injection_names = ["{}_{}".format(j, i) for i, j in
                       zip(compound_data_df['Compounds List'], compound_data_df['Injections'])]

    if not os.path.exists(pipeline_output_path): os.mkdir(pipeline_output_path)

    split_save_dir = os.path.join(pipeline_output_path, "all_timepoints_injections_split")
    if not os.path.exists(split_save_dir): os.mkdir(split_save_dir)

    for t in timepoints:
        print(t)
        raw_csv_file = data_files_dict[t]
        all_injections_timepoint, heads, mins,  = parse_agilent_raw_data.parse_agilent_raw_no_dict(raw_csv_file, timepoint=t, delimiter=',')


        parse_agilent_raw_data.save_individual_injections(all_data=all_injections_timepoint,
                                                          headers=heads,
                                                          minutes=mins,
                                                          save_directory=os.path.join(split_save_dir, str(t)),
                                                          injection_names = ["{}.csv".format(i) for i in injection_names],
                                                          raw_data_csv_path = raw_csv_file,
                                                          time_header='Timepoint')

    #Get the expected protein mw:
    theory_mass_protein = analyse.protein_mw_from_sequence(protein_sequence, oligo_state=protein_oligo_state)

    # Find, assign peaks + extract relative labelling
    analysed_data_path = os.path.join(pipeline_output_path, "analysed_injection_data")
    if not os.path.exists(analysed_data_path): os.mkdir(analysed_data_path)

    for i, i_name in enumerate(injection_names):
        injection_df_columns = ['Injection name', 'Compound name', 'Injection time', 'Peak mass', 'Error to closest theory mass (ppm)', 'Closest theory mass', 'Ligand count', 'Relative labelling']
        injection_df = pd.DataFrame(columns=injection_df_columns)

        peak_plotting_data = []
        timepoint_peak_data = []
        for t in timepoints:
            compound_smiles = compound_data_df['Smiles'][i]
            compound_mw_correction = compound_data_df['mass changes compare to MW of compounds'][i]
            compound_mol, compound_mw, corrected_compound_mw = analyse.get_molecule_data_from_smiles(compound_smiles,
                                                                                                     compound_mw_correction)

            # Read in the injection data
            input_csv = os.path.join(split_save_dir, str(t), '{}.csv'.format(i_name))
            ms_data_injection_df = pd.read_csv(input_csv)
            x_data = ms_data_injection_df['X(Daltons)'].values
            y_data = ms_data_injection_df['Y(Counts)'].values

            # Get only the values corresponding to the mass window
            mass_window_idxs = [np.where(x_data == m)[0][0] for m in mass_window]

            # Scale the data to an arbitraty number of counts that works with the thresholds
            scaled_y_data = np.divide(y_data, np.max(y_data))*50000

            window_scaled_y_data = scaled_y_data[mass_window_idxs[0]:mass_window_idxs[1]]
            window_x_data = x_data[mass_window_idxs[0]: mass_window_idxs[1]]

            peak_plotting_data.append(np.array([window_x_data, window_scaled_y_data]))
            analysed_peaks = analyse.assign_peaks(daltons=window_x_data,
                                                  scaled_counts=window_scaled_y_data,
                                                  theory_mw_protein=theory_mass_protein,
                                                  theory_mw_ligand=corrected_compound_mw,
                                                  prominence=peak_prominence,
                                                  correct_peak_positions=correct_peak_discretisation)

            labelling_percentages = analyse.calculate_relative_labelling(analysed_peaks, window_scaled_y_data, ppm_error)

            # # Visualise the ouputs:
            # single_timepoint_spectra = os.path.join(pipeline_output_path, 'single_timepoint_spectra')
            # if not os.path.exists(single_timepoint_spectra): os.mkdir(single_timepoint_spectra)
            # visualise.show_single_timepoint_peaks(analysed_peaks,
            #                                       window_x_data,
            #                                       window_scaled_y_data,
            #                                       labelling_percentages,
            #                                       single_timepoint_spectra,
            #                                       i_name,
            #                                       ms_data_injection_df['Timepoint'][0],
            #                                       50.00)

            for ap, lp in zip(analysed_peaks, labelling_percentages):
                df_list = [i_name, compound_data_df['Compounds List'][i],ms_data_injection_df['Timepoint'][0], ap[1], ap[2], ap[3], ap[4], lp]
                timepoint_peak_data.append(df_list)
            print('end timepoint {}, injection: {}'.format(str(t), i_name))

        timepoint_peak_data = np.array(timepoint_peak_data)
        print(timepoint_peak_data.shape)

        # TODO: catch error if df_dict not assigned
        for l in range(len(injection_df_columns)):
            injection_df[injection_df_columns[l]] = timepoint_peak_data[:, l]

        # write the data somewhere sensible
        injection_df_fname = os.path.join(analysed_data_path, "{}.csv".format(i_name))
        injection_df.to_csv(injection_df_fname)

        # Visualise the outputs
        # TODO: catch error if compound_smiles and corrected_compound_mw not assigned
        compound_summaries_dir = os.path.join(pipeline_output_path, "injection_summary_images")
        if not os.path.exists(compound_summaries_dir): os.mkdir(compound_summaries_dir)

        if display_mass_window:
            for d in range(len(peak_plotting_data)):
                data = peak_plotting_data[d]
                d_mass_window_idxs = [np.where(data[0] == m)[0][0] for m in display_mass_window]
                print(d_mass_window_idxs)
                print(data.shape)
                peak_plotting_data[d] = data[:, d_mass_window_idxs[0]:d_mass_window_idxs[1]]
                print(peak_plotting_data[d].shape)

        png_path = visualise.plot_compound_summary_new(compound_data=peak_plotting_data,
                                            injection_df_fname=injection_df_fname,
                                            expected_protein_mass=theory_mass_protein,
                                            compound_smiles=compound_smiles,
                                            corrected_compound_mass=corrected_compound_mw,
                                            save_dir=compound_summaries_dir)


if __name__ == '__main__':
    # Parameters needed to run the pipeline:
    raw_data_direc = "../Protein observe MS_raw/Data analysis/Processed raw data/Shipment 4 and 5 hits/200527/Raw data from Agilent/Full .csv dataset for each timepoint"
    data_files = {60: os.path.join(raw_data_direc, "1h all injections.csv"),
                  3: os.path.join(raw_data_direc, "200611 Moonshot 3 minutes.csv"),
                  180: os.path.join(raw_data_direc, "3H all injections.csv")}
    timepoints = sorted(list(data_files.keys()))

    compound_data_path = "../Protein observe MS_raw/200527 Shipment 4 and 5 compound list.csv"
    pipeline_output_path = 'pipeline_output'
    protein_seq_fasta = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"
    target_mass_window = [33500, 40000]
    plotting_mass_window = [33500, 35000]
    peak_prominence = 0.02
    ppm_error = 100.0

    main(data_files_dict=data_files,
         timepoints=timepoints,
         compound_data_path=compound_data_path,
         pipeline_output_path="new_outputs",
         protein_sequence=protein_seq_fasta,
         protein_oligo_state=1,
         mass_window=target_mass_window,
         peak_prominence=peak_prominence,
         ppm_error=ppm_error,
         display_mass_window=plotting_mass_window)
