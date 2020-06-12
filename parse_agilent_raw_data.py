import csv
import os

def parse_agilent_raw(raw_data_csv_path):
    """
    Parses the data from a an "all_injections.csv" file, automatically detects how many points in an injection.
    Assumes that each injection starts with a line similar to:
    '#"+ESI Scan <...>'
    :param raw_data_csv_path: string, path to the .csv file
    :return:
    """
    with open(raw_data_csv_path) as infile:
        infile.readline()
        reader = csv.DictReader(infile)

        all_injections=[]
        data = {}
        for row in reader:
            v = row[list(row.keys())[0]]
            # Detects a new injection
            if v is not None and v.startswith('#"'):
                all_injections.append(data)
                data = {}
                continue
            # Key fields autodetected by the csvreader. Do not want to add these lines to the injection data.
            elif v is not None and v.startswith('#'):
                continue
            for header, value in row.items():
                try:
                    data[header].append(value)
                except KeyError:
                    data[header] = [value]
        # Make sure the last injection gets appended as well:
        all_injections.append(data)
    return all_injections


def check_injection_data(injection_dict):
    """
    Does a rudimentary check that the data was processed ok.
    :param injection_dict: dictionary containing data for the injection. Expected keys:
    :return: string, a log message
    """
    out_string = ""
    # Check that all keys have the same number of values.
    lens = [len(v) for k, v in injection_dict.items()]
    try:
        assert (len(set(lens))) == 1
        out_string += "Number of points : {} \n".format(str(list(set(lens))[0]))
    except AssertionError:
        out_string += "Different numbers of readings for injection fields {} : \n".format(injection_dict.keys())
        for k in injection_dict.keys():
            out_string += "key: {0}, number readings: {1}, first value: {2}, last_value: {3} \n".format(
                k,
                len(injection_dict[k]),
                injection_dict[k][0],
                injection_dict[k][-1]
            )
    return out_string


def save_individual_injections(all_injections, save_directory, injection_names=None):
    """

    :param all_injections: List of dictionaries containing injection data. Output of parse_agilent_raw()
    :param save_directory: directory to save the split csvs to. WIll be created if doesn't exist.
    :param injection_names: optional (eg names of compounds + replicate number). If not supplied, the name will be the
    injection number, starting at 0.
    :return:
    """
    # Create save dir if doesn't exist already:
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    log_string='\n'

    if not injection_names:
        injection_names = ['injection_{}.csv'.format(x) for x in range(len(all_injections))]
    try:
        assert len(injection_names) == len(all_injections)
    except AssertionError:
        log_string += "Number of supplied injection names fo not match number of injections + \n"
        injection_names = ['injection_{}.csv'.format(str(x)) for x in range(len(all_injections))]

    for i in range(len(all_injections)):
        inj_dict = all_injections[i]

        # Check that the data is okay and log any problems
        injection_fname = os.path.join(save_directory, injection_names[i])
        check_str = check_injection_data(inj_dict)
        log_string += "{} \n".format(injection_names[i]) + check_str

        # Save injection data
        with open(injection_fname, 'w') as out_file:
            fieldnames = list(inj_dict.keys())
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            for k in range(len(inj_dict[fieldnames[0]])):
                writer.writerow({fieldnames[m]: inj_dict[fieldnames[m]][k] for m in range(len(fieldnames))})

    # Save the log file:
    log_fpath = os.path.join(save_directory, 'parser_log.txt')
    with open(log_fpath, 'w') as logfile:
        logfile.write(log_string)

if __name__ == "__main__":
    import pandas as pd

    compounds_file = os.path.join("Protein observe MS_raw", "200527 Shipment 4 and 5 compound list.csv")
    compounds_data = pd.read_csv(compounds_file)
    injection_names = ["{}_{}".format(j, i) for i, j in
                       zip(compounds_data['Compounds List'], compounds_data['Injections'])]

    raw_data_direc = "Protein observe MS_raw/Data analysis/Processed raw data/Shipment 4 and 5 hits/200527/Raw data from Agilent/Full .csv dataset for each timepoint"
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