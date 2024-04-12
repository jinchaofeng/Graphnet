import numpy as np
import torch


def get_DA(nodes, elements):
    ##adjacency matrix----------------------------------------------------
    # the number of nodes
    x_nodes, _ = np.shape(np.array(nodes))
    # initialize adjacency matrix
    A_hat = np.zeros([x_nodes, x_nodes])
    # store the node labels of each element
    mesh_ele = np.array(elements, dtype=np.int)
    # adjacency matrix without self-connection
    for i in range(x_nodes):
        ele_line, _ = np.where(mesh_ele == i + 1)
        s = mesh_ele[ele_line, :]
        ele_un = np.unique(s)
        # set the position of related nodes in adjacency matrix to 1
        A_hat[i, ele_un - 1] = 1
    # add self-connection into adjacency matrix
    for i in range(x_nodes):
        if A_hat[i, i] == 0:
            A_hat[i, i] = 1

    ##degree matrix----------------------------------------------
    D = np.diag(np.sum(A_hat, 0))
    # Degree matrix inversion
    D = np.linalg.inv(D)
    # D^-1*A=sDA
    sDA = np.dot(D, A_hat)
    A = torch.FloatTensor(sDA)
    return A
