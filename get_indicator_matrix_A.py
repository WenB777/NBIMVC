import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def get_mask(view_num, data_len, missing_rate):
    """Randomly generate incomplete multi-view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: e.g., 0.1, 0.3, 0.5, 0.7
        Returns:
          indicator matrix A

    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix    # indicator matrix A

# def get_mask(view_num, data_len, missing_rate):
#     """Randomly generate incomplete multi-view data.

#     Args:
#       view_num (int): Number of views.
#       data_len (int): Number of samples.
#       missing_rate (float): Missing rate, e.g., 0.1, 0.3, 0.5, 0.7.

#     Returns:
#       np.ndarray: Indicator matrix A.
#     """
#     # Adjust the missing rate per view
#     missing_rate /= view_num
#     one_rate = 1.0 - missing_rate

#     # Case when each sample has exactly one view preserved
#     if one_rate <= 1 / view_num:
#         enc = OneHotEncoder(sparse=False)
#         view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1)))
#         return view_preserve

#     # Case when all views are preserved
#     if one_rate == 1.0:
#         return np.ones((data_len, view_num), dtype=int)

#     # Iteratively adjust the ratio until error is within the tolerance
#     error = 1
#     tolerance = 0.005
#     target_ones = int(view_num * data_len * one_rate)
    
#     while error > tolerance:
#         # Generate a random binary mask with initial ratio close to one_rate
#         matrix = (randint(0, 100, size=(data_len, view_num)) < (one_rate * 100)).astype(int)
        
#         # Ensure that each sample has at least one view preserved
#         enc = OneHotEncoder(sparse=False)
#         view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1)))
#         matrix = np.maximum(matrix, view_preserve)

#         # Calculate the current ratio of 1s
#         current_ones = np.sum(matrix)
#         error = abs(target_ones - current_ones) / target_ones

#     return matrix