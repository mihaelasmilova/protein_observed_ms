from __future__ import print_function, division
import os
import logging
import numpy as np
import pandas as pd
import analyse
import visualise
import parse_agilent_raw_data
import yaml
import sys
import re


def main(config_file_path='config.yaml'):
    """
    :param config_file_path:
    :return:
    """
    
    if len(sys.argv)>2:
        config_file_path=sys.argv[1].strip()
    
    if not os.path.isfile(config_file_path):
        raise Exception('Configuration file not found at: '+config_file_path)
    
    with open(config_file_path) as f: #reading yaml file to yaml dict
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    

    #transferring yaml_dict params to function params
    data_files_dict=yaml_dict['raw_data_files']
    compound_data_path=yaml_dict['compound_data_path']
    
    if sys.platform=="win32":
        compound_data_path.replace(r"\\\\", r"\\")
        for v in data_files_dict.values():
            v.replace(r"\\\\", r"\\")
            
    timepoints = sorted(list(data_files_dict.keys()))
    pipeline_output_path=yaml_dict['pipeline_output_path']
    protein_sequence=yaml_dict['protein_sequence']
    protein_oligo_state=yaml_dict['protein_oligo_state']
    mass_window=yaml_dict['target_mass_window']
    peak_prominence=yaml_dict['peak_prominence']
    ppm_error=yaml_dict['ppm_error']
    display_mass_window=yaml_dict['plotting_mass_window']
    protein_mass=yaml_dict['protein_mass']
    correct_peak_discretisation=bool(yaml_dict['correct_peak_discretisation'])
    
       
    compound_data_df = pd.read_csv(compound_data_path)
    
    columns = compound_data_df.columns # Get the header names for the dataframe. The names don't matter, but the order does
    print('Expected column order in compound data spreadsheet:','(Compounds List)  (Injections)  (Smiles)  (Warhead type)  (Mass correction)')
    print('Supplied columns: {}'.format(columns.values))

    injection_names = ["{}_{}".format(j, i) for i, j in
                       zip(compound_data_df[columns[0]], compound_data_df[columns[1]])]
    
    for i in range(len(injection_names)): #clear spaces and :
        injection_names[i]=re.sub('\s+','-',injection_names[i].replace(':',''))
    
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
    if protein_mass == 0.0:
        theory_mass_protein = analyse.protein_mw_from_sequence(protein_sequence, oligo_state=protein_oligo_state)
    else:
        theory_mass_protein = protein_mass

    # Find, assign peaks + extract relative labelling
    analysed_data_path = os.path.join(pipeline_output_path, "analysed_injection_data")
    if not os.path.exists(analysed_data_path): os.mkdir(analysed_data_path)

    image_names = []
    df_master_list=[]
    for i, i_name in enumerate(injection_names):
        injection_df_columns = ['Injection name', 'Compound name', 'Injection time', 'Peak mass', 'Error to closest theory mass (ppm)', 'Closest theory mass', 'Ligand count', 'Relative labelling']
        injection_df = pd.DataFrame(columns=injection_df_columns)

        peak_plotting_data = []
        timepoint_peak_data = []
        for t in timepoints:
            compound_smiles = compound_data_df[columns[2]][i]
            compound_mw_correction = compound_data_df[columns[3]][i]
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
            x_axis_stagger = 1.0/15.0*(window_x_data[-1]-window_x_data[0])
            y_axis_stagger = 0.1*(max(scaled_y_data))

            peak_plotting_data.append(np.array([window_x_data, window_scaled_y_data]))
            analysed_peaks = analyse.assign_peaks(daltons=window_x_data,
                                                  scaled_counts=window_scaled_y_data,
                                                  theory_mw_protein=theory_mass_protein,
                                                  theory_mw_ligand=corrected_compound_mw,
                                                  prominence=peak_prominence,
                                                  correct_peak_positions=correct_peak_discretisation)

            labelling_percentages = analyse.calculate_relative_labelling(analysed_peaks, window_scaled_y_data, ppm_error)

            for ap, lp in zip(analysed_peaks, labelling_percentages):
                df_list = [i_name, compound_data_df['Compounds List'][i],ms_data_injection_df['Timepoint'][0], ap[1], ap[2], ap[3], ap[4], lp]
                timepoint_peak_data.append(df_list)
            print('end timepoint {}, injection: {}'.format(str(t), i_name))

        timepoint_peak_data = np.array(timepoint_peak_data)

        # TODO: catch error if df_dict not assigned
        for l in range(len(injection_df_columns)):
            injection_df[injection_df_columns[l]] = timepoint_peak_data[:, l]

        # write the data somewhere sensible
        injection_df_fname = os.path.join(analysed_data_path, "{}_peaks.csv".format(i_name))
        injection_df.to_csv(re.sub('\s+','-',injection_df_fname.replace(':','')))
        df_master_list.append(injection_df)

        # Visualise the outputs
        # TODO: catch error if compound_smiles and corrected_compound_mw not assigned
        compound_summaries_dir = os.path.join(pipeline_output_path, "injection_summary_images")
        if not os.path.exists(compound_summaries_dir): os.mkdir(compound_summaries_dir)

        if display_mass_window:
            for d in range(len(peak_plotting_data)):
                data = peak_plotting_data[d]
                d_mass_window_idxs = [np.where(data[0] == m)[0][0] for m in display_mass_window]
                peak_plotting_data[d] = data[:, d_mass_window_idxs[0]:d_mass_window_idxs[1]]

 
        png_path = visualise.plot_compound_summary_new(compound_data=peak_plotting_data,
                                            injection_df_fname=injection_df_fname,
                                            expected_protein_mass=theory_mass_protein,
                                            compound_smiles=compound_smiles,
                                            corrected_compound_mass=corrected_compound_mw,
                                            save_dir=compound_summaries_dir,
                                            x_axis_stagger=x_axis_stagger,
                                            y_axis_stagger=y_axis_stagger)
                                            
        image_names.append(png_path)
        
    with open(os.path.join(pipeline_output_path,'image_names.csv'), 'w') as f:
        f.write(',\n'.join(image_names))
    all_df=pd.concat(df_master_list)
    new_df=all_df.drop(all_df.columns[0],1)
    new_df.to_csv(os.path.join(pipeline_output_path,'all_injections_analysed.csv'),index=False)

if __name__ == '__main__':
                    
    main()
