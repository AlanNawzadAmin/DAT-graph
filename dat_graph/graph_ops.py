import numpy as np
from scipy.linalg import expm

def orient_v_structures(A_in, psis, skeleton, test_edges, moral_map, thresh_v_struct):
    """ Adds v structures to A_in. """
    n_nodes = len(A_in)
    for (i, j) in test_edges:
        # loop through pairs of nodes that are adjacent in the moral graph but not the skeleton
        if not skeleton[i, j]:
            # find both indices where adjacency of i, j was tested
            index_i = np.argwhere(np.all(test_edges == [i, j], axis=-1))[0, 0]
            index_j = np.argwhere(np.all(test_edges == [j, i], axis=-1))[0, 0]
            
            # get Markov blanket of i
            mb_i = moral_map[index_i]
            # label which elements in the MB are adjacent to both i and j in the skeleton
            not_intervention = np.logical_and(mb_i != -1, mb_i < n_nodes)
            skeleton_nbh_i = mb_i.astype(bool)
            skeleton_nbh_i[not_intervention] = (
                np.logical_and(skeleton[i, mb_i[not_intervention]],
                               skeleton[j, mb_i[not_intervention]]))
            skeleton_nbh_i[np.logical_not(not_intervention)] = False
    
            # same for j
            mb_j = moral_map[index_j]
            # label which elements in the MB are adjacent to both i and j in the skeleton
            not_intervention = np.logical_and(mb_j != -1, mb_j < n_nodes)
            skeleton_nbh_j = mb_j.astype(bool)
            skeleton_nbh_j[not_intervention] = (
                np.logical_and(skeleton[i, mb_j[not_intervention]],
                               skeleton[j, mb_j[not_intervention]]))
            skeleton_nbh_j[np.logical_not(not_intervention)] = False

            # get psis
            psi_score = (psis[index_i, skeleton_nbh_i] ** 2
                         + psis[index_j, skeleton_nbh_j] ** 2)
            # orient a v-structure if psis are small
            for collider in moral_map[index_i][skeleton_nbh_i][psi_score<thresh_v_struct]:
                A_in[collider, i] = 1
                A_in[collider, j] = 1
                A_in[i, collider] = 0
                A_in[j, collider] = 0
    return A_in

def meeks_rules(A_in, just_cycle=False):
    """If A is an adjacency matrix with uncertain edges at 0.5, assumes that
    triples are not v-structs and there are no cycles, and orients as such
    using Meek's rules."""
    A = np.copy(A_in)
    n_nodes = len(A)
    change = True
    while change:
        change = False
        if not just_cycle:
            # not v-struct
            for child, parent in np.argwhere(A == 1):
                other_par = np.argwhere(A[child] == 0.5)[:, 0]
                for v in other_par:
                    if A[parent, v] == 0 and A[v, parent] == 0:
                        A[child, v] = 0
                        A[v, child] = 1
                        change = True
    
        # no cycles
        dir_path = np.linalg.matrix_power((A == 1) + np.eye(n_nodes), n_nodes-1) > 0
        for child, parent in np.argwhere(dir_path * (A == 0.5)):
            A[child, parent] = 1
            A[parent, child] = 0
            change = True
    return A


def remove_edges_to_dag(A_in):
    """ Greedily remove edges from A until one reaches a DAG. """
    A = np.copy(A_in)
    n_edges_removed = 0
    while not get_perm(A)[0]:
        # find "loopiest" node
        i = np.argmax(np.diag(np.linalg.matrix_power(A+np.eye(len(A)), len(A)))-1)
        edges = np.array([[l, k] for l, k in np.argwhere(A) if l==i or k==i])
        # edges = np.argwhere(A)
        # greedily remove "loopiest" edge connected to that node
        eigs = np.empty(len(edges))
        for k, (i, j) in enumerate(edges):
            B = np.copy(A)
            B[i, j] = 0
            eigs[k] = (np.diag(np.linalg.matrix_power(B+np.eye(len(A)), len(A)))-1).sum()
        i, j = edges[np.argmin(eigs)]
        A[i, j] = 0
        n_edges_removed += 1
        print("Edges removed:", n_edges_removed)
    return A

def get_perm(A):
    """ 
    Parameters:
    A: np array
        Adjacency matrix.
    
    Returns:
    is_dag: bool
        True is A is a DAG
    perm: np array
        
    """
    n_pa = A.sum(axis=-1)
    start = np.argwhere(n_pa == n_pa.min())[:, 0]
    stack = start.tolist()
    substack = []
    seen = []
    perm = []
    while len(perm) + len(substack) < len(A):
        if not stack:
            # add leftover elements with smallest number of parents
            leftover = np.logical_not(np.isin(np.arange(len(A)), seen))
            n_pa_lo = n_pa[leftover]
            stack.extend(np.argwhere(leftover * (n_pa == n_pa_lo.min()))[:, 0].tolist())
        n = stack.pop()
        if not n in seen:
            seen.append(n)
            children = np.argwhere(A[:, n])[:, 0]
            stack.extend(children)

            while substack and A[n, substack[-1]] == 0:
                    perm.append(substack.pop())
            substack.append(n)
    perm = substack + perm[::-1]
    is_dag = np.all(np.triu(A[perm][:, perm]) == 0)
    assert np.all(np.sort(perm) == np.arange(len(A)))
    return is_dag, perm
