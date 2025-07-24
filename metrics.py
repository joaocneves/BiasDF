import numpy as np
import pandas as pd
from scipy.stats import stats

def ttest(data, gatt):

    """
    :param data:
    :param gatt: group attribute
    :return:
    """

    # Initialize a list to store the results
    differences = []

    # Define bin parameters
    n_bins = 100
    range_min = 0.0
    range_max = 1.0
    # Create bin edges
    bin_edges = np.linspace(range_min, range_max, n_bins + 1)

    X = data[data[gatt] == 1]
    NX = data[data[gatt] == 0]

    if not X.empty and not NX.empty:
        X_score = X['score'].mean()
        non_X_score = NX['score'].mean()
        score_diff = non_X_score - X_score

        H_X, _ = np.histogram(X['score'], bins=bin_edges, density=False)
        H_NX, _ = np.histogram(NX['score'], bins=bin_edges, density=False)

        H_X_prob = H_X / np.sum(H_X)
        H_NX_prob = H_NX / np.sum(H_NX)

        tpr_X = 1.0 - np.cumsum(H_X_prob)
        tpr_NX = 1.0 - np.cumsum(H_NX_prob)
        tpr_difference = np.mean(tpr_X - tpr_NX)

        differences.append({'X_score': X_score,
                            'non_X_score': non_X_score,
                            'score_difference': score_diff,
                            'tpr_X': tpr_X,
                            'tpr_NX': tpr_NX})

    # Convert the list of results into a DataFrame
    differences_df = pd.DataFrame(differences)

    # Perform the paired t-test
    t_stat, p_value = stats.ttest_ind(X['score'], NX['score'])

    # import matplotlib.pyplot as plt
    # plt.hist(differences_df['score_difference'], bins=200)
    # plt.show()

    return p_value

def paired_ttest(data, gatt, sgatt):

    """
    :param data:
    :param gatt: group attribute
    :param sgatt: subgroup attributes
    :return:
    """

    # Group the data based on the other attributes and then compare scores for males and non-males
    # Filter rows where both male and non-male records exist for the same attribute combination
    gdata = data.groupby(list(sgatt))

    # Initialize a list to store the results
    differences = []

    # Define bin parameters
    n_bins = 100
    range_min = 0.0
    range_max = 1.0
    # Create bin edges
    bin_edges = np.linspace(range_min, range_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Iterate through each group and calculate the score difference
    for _, group in gdata:
        X = group[group[gatt] == 1]
        NX = group[group[gatt] == 0]

        # If both male and non-male exist in the group, compute the difference in scores
        if not X.empty and not NX.empty:
            X_score = X['score'].mean()
            non_X_score = NX['score'].mean()
            score_diff = non_X_score - X_score

            H_X, _ = np.histogram(X['score'], bins=bin_edges, density=False)
            H_NX, _ = np.histogram(NX['score'], bins=bin_edges, density=False)

            H_X_prob = H_X / np.sum(H_X)
            H_NX_prob = H_NX / np.sum(H_NX)

            tpr_X = 1.0 - np.cumsum(H_X_prob)
            tpr_NX = 1.0 - np.cumsum(H_NX_prob)
            tpr_difference = np.mean(tpr_X-tpr_NX)

            differences.append({'attributes': group.iloc[0][sgatt].to_dict(),
                                'X_score': X_score,
                                'non_X_score': non_X_score,
                                'score_difference': score_diff,
                                'tpr_difference': tpr_difference})

    # Convert the list of results into a DataFrame
    differences_df = pd.DataFrame(differences)

    # Perform the paired t-test
    t_stat, p_value = stats.ttest_1samp(differences_df['tpr_difference'], 0)
    print(len(differences_df['tpr_difference']))

    # import matplotlib.pyplot as plt
    # plt.hist(differences_df['score_difference'], bins=200)
    # plt.show()

    return p_value


def eod(data, gatt ):

    """
    :param data:
    :param gatt: group attribute
    :param sgatt: subgroup attributes
    :return:
    """

    # Initialize a list to store the results
    differences = []

    # Define bin parameters
    n_bins = 100
    range_min = 0.0
    range_max = 1.0
    # Create bin edges
    bin_edges = np.linspace(range_min, range_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Iterate through each group and calculate the score difference

    X = data[data[gatt] == 1]
    NX = data[data[gatt] == 0]

    # If both male and non-male exist in the group, compute the difference in scores
    if not X.empty and not NX.empty:
        X_score = X['score'].mean()
        non_X_score = NX['score'].mean()
        score_diff = non_X_score - X_score

        H_X, _ = np.histogram(X['score'], bins=bin_edges, density=False)
        H_NX, _ = np.histogram(NX['score'], bins=bin_edges, density=False)

        H_X_prob = H_X / np.sum(H_X)
        H_NX_prob = H_NX / np.sum(H_NX)

        tpr_X = 1.0 - np.cumsum(H_X_prob)
        tpr_NX = 1.0 - np.cumsum(H_NX_prob)

        differences.append({'X_score': X_score,
                            'non_X_score': non_X_score,
                            'score_difference': score_diff,
                            'tpr_X': tpr_X,
                            'tpr_NX': tpr_NX})

    # Convert the list of results into a DataFrame
    differences_df = pd.DataFrame(differences)

    # Calculate the average of the curves
    tpr_D = np.mean(differences_df['tpr_X'] - differences_df['tpr_NX'])

    best_t = np.argmax(np.abs(tpr_D))

    eod_max = round(100 * tpr_D[best_t], 2)
    eod_mean = round(100 * np.mean(tpr_D[:-2]), 2)
    eod_std = round(100 * np.std(tpr_D[:-2]), 2)

    return eod_max, eod_mean, eod_std


def bias_risk(data, gatt, sgatt):

    """
    :param data:
    :param gatt: group attribute
    :param sgatt: subgroup attributes
    :return:
    """

    # Group the data based on the other attributes and then compare scores for males and non-males
    # Filter rows where both male and non-male records exist for the same attribute combination
    gdata = data.groupby(list(sgatt))

    # Initialize a list to store the results
    differences = []

    # Define bin parameters
    n_bins = 100
    range_min = 0.0
    range_max = 1.0
    # Create bin edges
    bin_edges = np.linspace(range_min, range_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Iterate through each group and calculate the score difference
    for _, group in gdata:
        X = group[group[gatt] == 1]
        NX = group[group[gatt] == 0]

        # If both male and non-male exist in the group, compute the difference in scores
        if not X.empty and not NX.empty:
            X_score = X['score'].mean()
            non_X_score = NX['score'].mean()
            score_diff = non_X_score - X_score

            H_X, _ = np.histogram(X['score'], bins=bin_edges, density=False)
            H_NX, _ = np.histogram(NX['score'], bins=bin_edges, density=False)

            H_X_prob = H_X / np.sum(H_X)
            H_NX_prob = H_NX / np.sum(H_NX)

            tpr_X = 1.0 - np.cumsum(H_X_prob)
            tpr_NX = 1.0 - np.cumsum(H_NX_prob)

            differences.append({'attributes': group.iloc[0][sgatt].to_dict(),
                                'X_score': X_score,
                                'non_X_score': non_X_score,
                                'score_difference': score_diff,
                                'tpr_X': tpr_X,
                                'tpr_NX': tpr_NX})

    # Convert the list of results into a DataFrame
    differences_df = pd.DataFrame(differences)

    # Calculate the average of the curves
    tpr_D = np.mean(differences_df['tpr_X'] - differences_df['tpr_NX'])

    best_t = np.argmax(np.abs(tpr_D))

    brisk_max = round(100 * tpr_D[best_t], 2)
    brisk_mean = round(100 * np.mean(tpr_D[:-2]), 2)

    return brisk_max, brisk_mean
