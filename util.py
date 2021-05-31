import numpy as np
from lingam.utils import make_dot


def save_adjacency_matrix_in_csv(file_name, adjacency_matrix, variable_names):
    """
    save the matrix in csv format with variable names
    """
    # create an empty matrix in object type (for string) with one extra row and column for variable names
    W_est_full_csv = np.array(np.zeros((adjacency_matrix.shape[0] + 1, adjacency_matrix.shape[1] + 1)), dtype=object)

    W_est_full_csv[1:, 1:] = adjacency_matrix  # copy adjacency matrix
    W_est_full_csv[0, 0] = 'column->row'
    W_est_full_csv[0, 1:] = variable_names  # set column names
    W_est_full_csv[1:, 0] = variable_names  # set row names

    np.savetxt(file_name + '.csv', W_est_full_csv, delimiter=',', fmt='%s')


def draw_DAGs_using_LINGAM(file_name, adjacency_matrix, variable_names):
    lower_limit = 0.0

    dot = make_dot(adjacency_matrix, labels=variable_names, lower_limit=lower_limit)

    dot.format = 'png'
    dot.render(file_name)
