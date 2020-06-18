from __future__ import print_function, division
import csv
import os
import logging


def parse_agilent_raw_no_dict(raw_data_csv_path, delimiter=',', timepoint=0.0):
    """
    Parses the data from a an "all_injections.csv" file, automatically detects how many points in an injection.
    Assumes that each injection starts with a line similar to:
    '#"+ESI Scan <...>'
    :param raw_data_csv_path: string, path to the .csv file
    :return:
    """
    with open(raw_data_csv_path) as infile:
        # Holds the data for all the injections in the csv
        all_data = []
        # Data for an individual injection
        injection_data = []
        # The names of the fields, usually something like :'# Point', 'X(Daltons)', 'Y(Counts)'
        headers = []
        # The actual times of the injections (relative to the timepoint supplied)
        minutes = []

        for i, line in enumerate(infile.readlines()):
            # The headers containing time information look like  '#"+ESI Scan <...>'
            if line.startswith('#"'):
                if len(minutes) == 0:
                    first_timepoint_file = float(line.split('-')[-1].split(' ')[0])
                minute = float(line.split('-')[-1].split(' ')[0]) - first_timepoint_file + timepoint
                minutes.append(round(minute, 2))

                if i > 0:
                    all_data.append(injection_data)
                    injection_data = []

            # The field names lines start with '#Point'
            elif line.startswith('#'):
                headers.append(line.strip().split(delimiter))
            # Anything else is assumed to be data
            else:
                injection_data.append(line.strip().split(delimiter))

        all_data.append(injection_data)

    if len(minutes) == len(headers) == len(all_data):
        return all_data, headers, minutes
    else:
        error_msg = "Inconsistency between number of injections, starting times and headers in file {}:\n" \
                    "Number injections detected: {} \n" \
                    "Number timepoints detected: {} \n" \
                    "Number of field headers detected:{} \n".format(raw_data_csv_path,
                                                              len(all_data),
                                                              len(minutes),
                                                              len(headers))
        logging.error(msg=error_msg)
        raise ValueError(error_msg)


def save_individual_injections(all_data,
                               headers,
                               minutes,
                               save_directory,
                               injection_names=None,
                               raw_data_csv_path=None,
                               time_header='Timepoint'):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    if not injection_names:
        injection_names = ['injection_{}.csv'.format(x) for x in range(len(all_data))]
    try:
        assert len(injection_names) == len(all_data)

    except AssertionError:
        log_string = "Number of supplied injection names do not match number of injections in file {} + \n".format(
            raw_data_csv_path)
        logging.error(msg=log_string)

        injection_names = ['injection_{}.csv'.format(str(x)) for x in range(len(all_data))]

    for i in range(len(all_data)):
        injection_name = injection_names[i]
        injection_fname = os.path.join(save_directory, injection_name)
        injection_data = all_data[i]
        column_names = headers[i]
        corrected_time = minutes[i]

        # Make sure that the number of columns in the data is consistent
        m = [len(injection_data[j]) for j in range(len(injection_data))]

        if len(set(m)) != 1:
            column_num_error = """Inconsistent number of detected columns in data for injection {} in file {} \n
            Number of columns detected: {} \n
            This injection will be processed, but donwstream analysis may not work/be trustworthy\n""".format(
                injection_name,
                raw_data_csv_path,
                str(set(m)))
            logging.error(msg=column_num_error)

        if list(set(m))[0] != len(column_names):
            column_names_num_error = """Detected number of data columns in injection {} in file {} is consistent, but the number of detected column names doesn't match. \n
            Detected number of columns: {} \n
            Detected column names: {} \n
            This injection will be processed, but donwstream analysis may not work/be trustworthy\n""".format(
                injection_name,
                raw_data_csv_path,
                str(set(m)),
                str(column_names))

        column_names.append(time_header)

        for k in injection_data:
            k.append(corrected_time)
        all_injection_data_file = [column_names] + injection_data

        with open(injection_fname, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_injection_data_file)


if __name__ == "__main__":
    import pandas as pd

    compounds_file = os.path.join("../Protein observe MS_raw", "200527 Shipment 4 and 5 compound list.csv")
    compounds_data = pd.read_csv(compounds_file)
    injection_names = ["{}_{}".format(j, i) for i, j in
                       zip(compounds_data['Compounds List'], compounds_data['Injections'])]

    raw_data_direc = "../Protein observe MS_raw/Data analysis/Processed raw data/Shipment 4 and 5 hits/200527/Raw data from Agilent/Full .csv dataset for each timepoint"
    data_files = {'1h': "1h all injections.csv",
                  '3m': "200611 Moonshot 3 minutes.csv",
                  '3h': "3H all injections.csv"}


    split_save_dir = "all_timepoints_injections_split"
    if not os.path.exists(split_save_dir): os.mkdir(split_save_dir)

    for timepoint in data_files.keys():
        all_injections_timepoint = parse_agilent_raw(os.path.join(raw_data_direc, data_files[timepoint]))
        save_individual_injections(all_injections_timepoint,
                                   os.path.join(split_save_dir, timepoint),
                                   ["{}.csv".format(i) for i in injection_names])