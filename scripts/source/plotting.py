import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import vector

plt.style.use(hep.style.CMS)


def get_bin_center(bins):
    #returns center of bins
    data_bins_left = np.copy(bins[:-1]) /2
    data_bins_right = np.copy(bins[1:]) /2
    bin_center = data_bins_left + data_bins_right
    return bin_center



def control_plot(data_col, emb_col, bins, title, dy=None):
    """plots data and embedding in one histogram and their ratio in a separate plot"""
    #creating figure, selecting upper axis 
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.set_figheight(14)
    fig.set_figwidth(14)
    plt.sca(ax[0])
    ax_temp = ax[0]

    #plotting embeddign histogram
    emb_hist, edges, _ = ax_temp.hist(emb_col, bins, label=r"$\mu\rightarrow\mu$ embedding", histtype="stepfilled")
    ax_temp.hist(emb_col, edges, histtype="step", color="black", linewidth=1.5)

    #plotting data points and errorbars
    data_hist, _ = np.histogram(data_col, edges)
    bins_data_center = get_bin_center(edges)
    data_errors = np.sqrt(data_hist)
    ax_temp.errorbar(bins_data_center, data_hist, xerr=np.diff(edges)/2, yerr=data_errors, label="data", c="black", fmt="o", linestyle="none", markersize=8)

    #adding title, labels and legend to upper plot
    ax_temp.set_yscale("log")
    ax_temp.set_ylabel(r"N$_\text{events}$")
    hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8
    plt.subplots_adjust(hspace=0.05)

    legend = plt.legend(loc="upper right", markerfirst=False)
    for handle in legend.get_patches():
        handle.set_edgecolor("black")  # Add black border to legend symbol
        handle.set_linewidth(1.5)  # Make edge visible

    #selecting lower figure
    plt.sca(ax[1])
    ax_temp = ax[1]

    rel_diff = divide_arrays(data_hist, emb_hist)
    rel_diff_error = divide_arrays(data_errors, emb_hist)

    ax_temp.errorbar(bins_data_center, rel_diff, xerr=np.diff(edges)/2, yerr=rel_diff_error, label="observed", c="black", fmt="o", linestyle="none", markersize=8)
    ax_temp.bar(bins_data_center, 2*rel_diff_error, width=np.diff(edges), bottom=1-rel_diff_error, color="grey", alpha=0.5, edgecolor="none")

    #adding label and helping line to plot
    if type(dy)==type(None):
        dy_max = get_dy_max(rel_diff-1)
    else:
        dy_max = dy
        
    ax_temp.set_ylim(1-dy_max, 1+dy_max)
    ax_temp.set_ylabel(r"$N_\text{data}/N_\text{emb}$")
    plt.axhline(y=1, xmin=0, xmax=1, linestyle="dashed", color="black")
    ax_temp.set_xlabel(title)

    return ax

def divide_arrays(numerator, denominator):
    #divides numerator by denominator while avoiding division by zero errors
    
    result = np.full_like(numerator, np.nan, float)
    nan_mask = np.logical_and(~np.isnan(denominator), ~np.isnan(numerator))
    non_zero_mask = denominator!=0 
    mask = np.logical_and(nan_mask, non_zero_mask)

    result[mask] = numerator[mask]/ denominator[mask]

    return result

def histogram(quantity, bins, title):
    """subtracts both columns from each other and plots them in a histogram"""

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(14)
    fig.set_figwidth(14)

    hist, edges, _ = ax.hist(quantity, bins, histtype="stepfilled")
    ax.hist(quantity, edges, histtype="step", color="black", linewidth=1.5)

    ax.set_ylabel(r"N$_\text{events}$")
    ax.set_yscale("log")
    ax.set_xlabel(title)
    hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8

    # legend = plt.legend(loc="upper right", markerfirst=False)
    # for handle in legend.get_patches():
    #     handle.set_edgecolor("black")  # Add black border to legend symbol
    #     handle.set_linewidth(1.5)  # Make edge visible

    return ax



def q_comparison(col1, col2, bins, col1_label, col2_label, title):
    #compares two columns against each other and also shows ratio

    #creating figure, selecting upper axis 
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.set_figheight(14)
    fig.set_figwidth(14)
    plt.sca(ax[0])
    ax_temp = ax[0]
    ax_temp.set_yscale("log")

    #plotting corrected embeddign histogram
    hist1, edges, _ = ax_temp.hist(col1, bins, label=col1_label, histtype="step", linewidth=2)

    #plotting uncorrected embeddign histogram
    hist2, _, _ = ax_temp.hist(col2, edges, label=col2_label, histtype="step", linewidth=2)

    # master, _, _ = ax_temp.hist(master_col, edges, label=r"Unmatched", histtype="step", linewidth=2)


    #adding title, labels and legend to upper plot
    ax_temp.set_ylabel(r"N$_\text{events}$")
    hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8
    plt.subplots_adjust(hspace=0.05)
    plt.legend()

    #selecting lower figure
    plt.sca(ax[1])
    ax_temp = ax[1]

    rel_difference = divide_arrays(hist1, hist2)
    rel_difference_errors = divide_arrays(np.sqrt(hist1), hist2)
    bins_data_center = get_bin_center(edges)

    # ax_temp.hist(rel_difference, bins, histtype="step", linewidth=2, color="black")
    ax_temp.step(edges[:-1], rel_difference, where="post", color="black")
    ax_temp.bar(bins_data_center, 2*rel_difference_errors, width=np.diff(edges), bottom=rel_difference-rel_difference_errors, color="grey", alpha=0.5, edgecolor="none")

    dy_max = get_dy_max(rel_difference-1)
    ax_temp.set_ylim(1-dy_max, 1+dy_max)

    #adding label and helping line to plot
    ax_temp.set_ylabel(r"Unmatched / Matched")
    plt.axhline(y=1, xmin=0, xmax=1, linestyle="dashed", color="darkgrey", linewidth=1)
    ax_temp.set_xlabel(title)

    return ax


def get_dy_max(array):
    #returns ylim for error plots
    not_nan_array = array[~np.isnan(array)]
    array_max = np.amax(np.absolute(not_nan_array))
    dy_max = min(1, 1.5*array_max)#maximum of dy has to be 1
    dy_max = max(dy_max, 0.1)#minimum only 10%
    if dy_max == 0:
        return 1
    return dy_max


def x_vs_y(x, y, xlabel, ylabel):
    #plots x as funciton of y
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(14)
    fig.set_figwidth(14)

    ax.scatter(x, y)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8

    # legend = plt.legend(loc="upper right", markerfirst=False)
    # for handle in legend.get_patches():
    #     handle.set_edgecolor("black")  # Add black border to legend symbol
    #     handle.set_linewidth(1.5)  # Make edge visible

    return ax


def nq_comparison(q_dict, bins, title, data=None):
    #function for creating plots with variable amount of dataset

    #creating figure, selecting upper axis 
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(14)
    fig.set_figwidth(14)


    for label in q_dict:
        col = q_dict[label]

        #plotting corrected embeddign histogram
        _, bins, _ = ax.hist(col, bins, label=label, histtype="step", linewidth=2)

    if type(data)!=type(None):
        data_hist, _ = np.histogram(data, bins)
        bins_data_center = get_bin_center(bins)
        data_errors = np.sqrt(data_hist)
        ax.errorbar(bins_data_center, data_hist, xerr=np.diff(bins)/2, yerr=data_errors, label="Data", c="black", fmt="o", linestyle="none", markersize=8)

    ax.set_xlabel(title)
    #adding title, labels and legend to upper plot
    hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8
    plt.subplots_adjust(hspace=0.05)
    plt.legend()


    return ax


def match_plot(fit, title):
    #function to be used for creating a hsitogram indicating the best fit for leading and trailing muon
    best_fit1 = fit[:,0]
    best_fit2 = fit[:,1]
    max_id = int(np.nanmax(fit))
    ticks = [n for n in range(-1, max_id+1)]
    tick_labels = [np.nan] + [n for n in range(max_id+1)]

    best_fit1 = np.where(np.isnan(best_fit1), -1, best_fit1)
    best_fit2 = np.where(np.isnan(best_fit2), -1, best_fit2)

    max_id = ticks[-1] + 1.5
    ax = nq_comparison({"LM":best_fit1, "TM":best_fit2}, np.arange(-1.5, max_id, 1), title)
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    return ax

def dr_plot(dr, title):
    #plotting dr distributions
    dr1 = dr[:,0]
    dr2 = dr[:,1]

    ax = nq_comparison({"LM":dr1, "TM":dr2}, 30, title)

    return ax
