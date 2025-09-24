import uproot
import os
import pandas as pd
import numpy as np
import vector
import matplotlib.pyplot as plt

def set_plt_fonts():
    SMALL_SIZE = 30
    MEDIUM_SIZE = 35
    # BIGGER_SIZE = 12

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def print_stats(n_match, nlone_data, nlone_emb):
    print("Number of events:                    ", n_match.shape)
    print("Number of unm. obj. in data:         ", np.sum(nlone_data))
    print("Number of unm. obj. in emb:          ", np.sum(nlone_emb))
    print("Number of m. obj. in emb:            ", np.sum(n_match))
    print("events with no matchable obj.:       ", np.sum(n_match==0))
    print("events without unmatchable obj. data:", np.sum(nlone_data==0))
    print("events without unmatchable obj. emb: ", np.sum(nlone_emb==0))
    print("events with unmatchable obj. data:   ", np.sum(nlone_data>0))
    print("events with unmatchable obj. emb:    ", np.sum(nlone_emb>0))
    print("events with error in emb/ data:      ", np.logical_or(nlone_data>0, nlone_emb>0).sum())
    mask = np.logical_and(nlone_data==0, nlone_emb==0)
    print("events without error in emb/ data:   ", mask.sum())
    mask = np.logical_and(mask, n_match>0)
    print("events w. obj. and w/o unm. obj.:    ", mask.sum(), "\n\n")

def print_zmumu_stats(dist_data, dist_emb, cut):

    mask = dist_data<cut
    print(f"Number of data objects left from cut: ", mask.sum())
    print(f"Number of events left from cut:       ", mask.any(axis=1).sum())
    mask = dist_data>cut
    print(f"Number of data objects right from cut:", mask.sum())
    print(f"Number of events right from cut:       ", mask.any(axis=1).sum())
    mask = dist_emb<cut
    print(f"Number of emb objects left from cut:  ", mask.sum())
    print(f"Number of events left from cut:       ", mask.any(axis=1).sum())
    mask = dist_emb>cut
    print(f"Number of emb objects right from cut: ", mask.sum())
    print(f"Number of events right from cut:       ", mask.any(axis=1).sum(), "\n\n")


def detect_changes(df1, df2, columns:list):
    #compares how many elements in the series object are different between two dfs
    res = ""
    for column in columns:
        temp = df1[column] - df2[column]
        mask1 = ~np.isnan(temp)
        mask2 = temp != 0
        mask = np.logical_and(mask1, mask2)
        count = mask.sum()
        res += f"{column}: {count}; "
    print(res + "rows different")



def subtract_columns(col1, col2, col_name:str):
    #allows to subtract two columns while treating phi specially
    if not "phi" in col_name:
        diff = np.abs(col1 - col2)
    #phi needs to be handled differently because the value must be lower than pi
    else:
        diff = np.abs(col1 - col2)

        mask = diff > np.pi
        
        diff[mask] = 2*np.pi - diff[mask]

    return diff


def divide_columns(numerator, divisor):
    #divides columns wile avoiding dividing by zero warnings and nan errors
    mask1 = divisor != 0
    mask2 = ~np.isnan(divisor)
    mask3 = ~np.isnan(numerator)

    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)

    q = np.full_like(numerator, np.nan)
    q[mask] = numerator[mask]/ divisor[mask]

    return q

def get_matching_df(df, rm_cols):
    #this function copies a dataframe so that the original data stays untouched and also removes columns that are unwanted in the resulting dataset
    df_copy = df.copy(deep=True)
    for col in rm_cols:
        del df_copy[col]

    return df_copy

def copy_column_set(from_df, to_df, basename):

    for column in from_df.columns:
        if column.startswith(basename):
            new_name = column.replace(basename, basename+"cp_")
            to_df[new_name] = from_df[column].copy(deep=True)

    return to_df

# def prepare_jet_matching(data, emb):

#     data["LJ_pt"] = data["Jet_pt_1"].copy(deep=True)
#     data["TJ_pt"] = data["Jet_pt_2"].copy(deep=True)
#     data["LJ_eta"] = data["Jet_eta_1"].copy(deep=True)
#     data["TJ_eta"] = data["Jet_eta_2"].copy(deep=True)
#     data["LJ_phi"] = data["Jet_phi_1"].copy(deep=True)
#     data["TJ_phi"] = data["Jet_phi_2"].copy(deep=True)
#     data["LJ_m"] = data["Jet_m_1"].copy(deep=True)
#     data["TJ_m"] = data["Jet_m_2"].copy(deep=True)

#     emb_for_matching = emb[["run", "lumi", "event"]].copy(deep=True)

#     for column in emb.columns:
#         if column.startswith("Jet_"):
#             emb_for_matching[column] = emb[column].copy(deep=True)

#     emb_for_matching["LJ_pt"] = data["Jet_pt_1"].copy(deep=True)
#     emb_for_matching["TJ_pt"] = data["Jet_pt_2"].copy(deep=True)
#     emb_for_matching["LJ_eta"] = data["Jet_eta_1"].copy(deep=True)
#     emb_for_matching["TJ_eta"] = data["Jet_eta_2"].copy(deep=True)
#     emb_for_matching["LJ_phi"] = data["Jet_phi_1"].copy(deep=True)
#     emb_for_matching["TJ_phi"] = data["Jet_phi_2"].copy(deep=True)
#     emb_for_matching["LJ_m"] = data["Jet_m_1"].copy(deep=True)
#     emb_for_matching["TJ_m"] = data["Jet_m_2"].copy(deep=True)

#     return data, emb_for_matching


def prepare_matching(data, emb, mode):
    if mode == "electron":
        short = "E"
        basename = "Electron"
    elif mode == "photon":
        short = "P"
        basename = "Photon"
    elif mode == "jet":
        short = "J"
        basename = "Jet"

    data[f"L{short}_pt"] = data[f"{basename}_pt_1"].copy(deep=True)
    data[f"T{short}_pt"] = data[f"{basename}_pt_2"].copy(deep=True)
    data[f"L{short}_eta"] = data[f"{basename}_eta_1"].copy(deep=True)
    data[f"T{short}_eta"] = data[f"{basename}_eta_2"].copy(deep=True)
    data[f"L{short}_phi"] = data[f"{basename}_phi_1"].copy(deep=True)
    data[f"T{short}_phi"] = data[f"{basename}_phi_2"].copy(deep=True)

    if mode != "photon":
        data[f"L{short}_m"] = data[f"{basename}_m_1"].copy(deep=True)
        data[f"T{short}_m"] = data[f"{basename}_m_2"].copy(deep=True)

    emb_for_matching = emb[["run", "lumi", "event"]].copy(deep=True)

    for column in emb.columns:
        if column.startswith(basename):
            emb_for_matching[column] = emb[column].copy(deep=True)

    emb_for_matching[f"L{short}_pt"] = data[f"{basename}_pt_1"].copy(deep=True)
    emb_for_matching[f"T{short}_pt"] = data[f"{basename}_pt_2"].copy(deep=True)
    emb_for_matching[f"L{short}_eta"] = data[f"{basename}_eta_1"].copy(deep=True)
    emb_for_matching[f"T{short}_eta"] = data[f"{basename}_eta_2"].copy(deep=True)
    emb_for_matching[f"L{short}_phi"] = data[f"{basename}_phi_1"].copy(deep=True)
    emb_for_matching[f"T{short}_phi"] = data[f"{basename}_phi_2"].copy(deep=True)

    if mode != "photon":
        emb_for_matching[f"L{short}_m"] = data[f"{basename}_m_1"].copy(deep=True)
        emb_for_matching[f"T{short}_m"] = data[f"{basename}_m_2"].copy(deep=True)

    return data, emb_for_matching

def verify_events(*data):
    #checks whether all dataframes have the same order
    master_df = data[0]

    for df in data:
        compare_cells(master_df["event"].values, df["event"].values)
        compare_cells(master_df["lumi"].values, df["lumi"].values)
        compare_cells(master_df["run"].values, df["run"].values)


def compare_cells(column1, column2):
    #raises assertion if column1 deviates from column2
    test = column1 - column2
    assert len(test[test!=0]) == 0, "Mismatch"


def create_concordant_subsets(df1, df2):
    #performs inner merge on keys from df1 and df2. dataframes will have the same length afterwards
    l1 = len(df1)
    l2 = len(df1)

    keys = ["run", "lumi", "event"]
    df1 = df1.sort_values(by=keys, ignore_index=True)
    df2 = df2.sort_values(by=keys, ignore_index=True)

    mask = df1[keys].merge(df2[keys], how="inner")

    l3 = len(mask)
    # print(f"Previous lengths: {l1}, {l2} - New length: {l3}")

    df1 = df1.merge(mask, how="inner")
    df2 = df2.merge(mask, how="inner")

    return df1, df2

def copy_columns_from_to(from_df, to_df, columns):
    #copies columns from from_df to to_df
    cols_to_copy = from_df[columns].copy(deep=True)
    to_df[columns] = cols_to_copy

    return to_df


def set_working_dir():
    #sets working dir to the validation suite folder so that jupyter notebooks and scripts behave the same and the paths do not need to be altered

    path = os.getcwd()

    assert "validation_suite" in path, "You renamed the folder without my permission. "

    folders = path.split("/")

    new_path = ""

    for element in folders:
        if element != "validation_suite":
            new_path += element + "/"
        else:
            new_path += element + "/"
            break
    #setting the working directory to the validation suite folder
    os.chdir(new_path)

    path = os.getcwd()#directory of jupyter notebook
    print(f"Working dir: {path}")


def col_is_expanded(quantities, col):
    # checks in importing instructions whether a column is being expanded or not
    for q in quantities:
        if q["target"] == col:#q["target"] contains the column name 
            return q["expand"]#and this is the variable for setting expansion rules
        
    raise ValueError("Column not found")



def get_n_occurence(df, basename):
    #returns the occurence of a column. pt_1, pt_2 is counted as pt
    n = 0
    for col in df.columns: 
        if col.startswith(basename):
            n += 1

    return n

def count_n_objects(df, col):
    subset = df[[c for c in df.columns if c.startswith(col)]]
    counts = subset.notna().sum(axis=1)
    
    return counts