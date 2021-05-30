import numpy as np
from lingam.utils import make_dot


def save_adjacency_matrix_in_csv(file_name, adjacency_matrix, variable_names):
    """
    save the matrix in csv format with variable names
    """
    # create an empty matrix in object type (for string) with one extra row and column for variable names
    W_est_full_csv = np.array(np.zeros((adjacency_matrix.shape[0] + 1, adjacency_matrix.shape[1] + 1)), dtype=object)

    W_est_full_csv[1:, 1:] = adjacency_matrix  # copy adjacency matrix
    W_est_full_csv[0, 0] = 'row->column'
    W_est_full_csv[0, 1:] = variable_names  # set column names
    W_est_full_csv[1:, 0] = variable_names  # set row names

    np.savetxt(file_name + '.csv', W_est_full_csv, delimiter=',', fmt='%s')


def draw_DAGs_using_LINGAM(file_name, adjacency_matrix, variable_names):
    # direction of the adjacency matrix needs to be transposed.
    # in LINGAM, the adjacency matrix is defined as column variable -> row variable
    # in NOTEARS, the W is defined as row variable -> column variable

    # the default value here was 0.01. Instead of not drawing edges smaller than 0.01, we eliminate edges
    # smaller than `w_threshold` from the estimated graph so that we can set the value here to 0.
    lower_limit = 0.0

    dot = make_dot(np.transpose(adjacency_matrix), labels=variable_names, lower_limit=lower_limit)

    dot.format = 'png'
    dot.render(file_name)
