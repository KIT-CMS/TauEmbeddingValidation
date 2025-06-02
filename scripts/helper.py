import uproot
import os
import pandas as pd
import numpy as np
import vector


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

def prepare_jet_matching(data, emb):

    data["LJ_pt"] = data["Jet_pt_1"].copy(deep=True)
    data["TJ_pt"] = data["Jet_pt_2"].copy(deep=True)
    data["LJ_eta"] = data["Jet_eta_1"].copy(deep=True)
    data["TJ_eta"] = data["Jet_eta_2"].copy(deep=True)
    data["LJ_phi"] = data["Jet_phi_1"].copy(deep=True)
    data["TJ_phi"] = data["Jet_phi_2"].copy(deep=True)
    data["LJ_m"] = data["Jet_m_1"].copy(deep=True)
    data["TJ_m"] = data["Jet_m_2"].copy(deep=True)

    emb["LJ_pt"] = data["Jet_pt_1"].copy(deep=True)
    emb["TJ_pt"] = data["Jet_pt_2"].copy(deep=True)
    emb["LJ_eta"] = data["Jet_eta_1"].copy(deep=True)
    emb["TJ_eta"] = data["Jet_eta_2"].copy(deep=True)
    emb["LJ_phi"] = data["Jet_phi_1"].copy(deep=True)
    emb["TJ_phi"] = data["Jet_phi_2"].copy(deep=True)
    emb["LJ_m"] = data["Jet_m_1"].copy(deep=True)
    emb["TJ_m"] = data["Jet_m_2"].copy(deep=True)

    return data, emb

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
    print(f"Previous lengths: {l1}, {l2} - New length: {l3}")

    df1 = df1.merge(mask, how="inner")
    df2 = df2.merge(mask, how="inner")

    return df1, df2

def copy_columns_from_to(from_df, to_df, columns):
    #copies columns from from_df to to_df
    cols_to_copy = from_df[columns].copy(deep=True)
    to_df[columns] = cols_to_copy

    return from_df, to_df
