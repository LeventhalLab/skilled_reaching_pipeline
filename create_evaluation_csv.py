import pickle
import glob
import os
import pandas as pd


def read_pickle(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def convert_full_pickle_to_pandas(full_pickle):
    pass


def read_learning_stats(filename, column_headers=('iteration', 'loss', 'scmap loss', 'locref loss', 'limb loss', 'lr')):
    lstats = pd.read_csv(filename, header=None)

    lstats_dict = {col_header: [] for col_header in column_headers}
    for i_col, col_name in enumerate(column_headers):
        col_data = extract_values_from_column(col_name, i_col, lstats)
        lstats_dict[col_name] = col_data

    lstats_df = pd.DataFrame.from_dict(lstats_dict)

    return lstats_df


def write_learning_stats(filename, lstats_df):
    lstats_df.to_csv(filename)
    pass


def extract_values_from_column(column_name, column_number, lstats):
    lstats_col = lstats[column_number]

    num_rows = len(lstats_col)
    label_length = len(column_name) + 2
    return_data = []
    for i_row in range(num_rows):
        # extract current value from this row

        try:
            return_data.append(float(lstats_col[i_row][label_length:]))
        except:
            pass

    return return_data


if __name__ == '__main__':

    full_name = '/Users/dan/Documents/deeplabcut/skilled_reaching_direct-Dan_Leventhal-2020-10-19/evaluation-results/iteration-0/skilled_reaching_directOct19-trainset95shuffle1/DLC_resnet50_skilled_reaching_directOct19shuffle1_200000-snapshot-200000_full.pickle'
    meta_name = '/Users/dan/Documents/deeplabcut/skilled_reaching_direct-Dan_Leventhal-2020-10-19/evaluation-results/iteration-0/skilled_reaching_directOct19-trainset95shuffle1/DLC_resnet50_skilled_reaching_directOct19shuffle1_200000-snapshot-200000_meta.pickle'
    lstats_name ='/Users/dan/Documents/deeplabcut/Kat_results/train/learning_stats.csv'
    # lstats_save_name = '/Users/dan/Documents/deeplabcut/Kat_results/train/learning_stats_new.csv'
    lstats_dir = '/Users/dan/Documents/deeplabcut/Kat_training_CSVs'

    lstats_name_list = glob.glob(lstats_dir + '/*.csv')

    for lstats_file in lstats_name_list:

        _, lstats_tail = os.path.split(lstats_file)
        lstats_name, _ = os.path.splitext(lstats_tail)
        new_name = lstats_name + '_new.csv'
        lstats_save_name = os.path.join(lstats_dir, new_name)

        lstats_df = read_learning_stats(lstats_file)
        write_learning_stats(lstats_save_name, lstats_df)

    # full_pickle = read_pickle(full_name)
    # meta_pickle = read_pickle(meta_name)

    pass