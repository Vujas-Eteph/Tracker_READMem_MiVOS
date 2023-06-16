# Create permutation matrices using a greedy approach by looking only at a 2D Matrix that is squared
#
#

import numpy as np
import torch
import time
from functools import wraps
from munkres import Munkres, print_matrix, make_cost_matrix
import math
import scipy



def khronos(func):
    @wraps(func)
    def khronos_metron_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # Return to a seperate log, so that this can be processed at an ulterior stage, by a diff. prog.
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return khronos_metron_wrapper

@khronos
def Hungarian_permutation(square_matrix_Tensor):
    square_matrix = square_matrix_Tensor.clone().detach().cpu().numpy()
    permutation_matrix = torch.zeros([*square_matrix_Tensor.shape])
    cost_matrix = make_cost_matrix(square_matrix) # I want to maximize the profit of the matrix
    m = Munkres()
    indexes = m.compute(cost_matrix)

    indexes_np = np.array(indexes)
    idx = indexes_np[:, 0] # rows
    jdx = indexes_np[:, 1] # columns

    permutation_matrix[idx,jdx] = 1.0

    return permutation_matrix

@khronos
def convert_2_cost_matrix(square_matrix):
    return square_matrix.max() - square_matrix


@khronos
def scipy_help_for_permutation(square_matrix_Tensor):
    square_matrix = square_matrix_Tensor.clone().detach().cpu().numpy()
    permutation_matrix = torch.zeros([*square_matrix_Tensor.shape])
    # cost_matrix = convert_2_cost_matrix(square_matrix) # I want to maximize the profit of the matrix
    # idx, jdx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
    idx, jdx = scipy.optimize.linear_sum_assignment(square_matrix, maximize=True)
    permutation_matrix[idx,jdx] = 1.0

    return permutation_matrix



# @khronos
# def create_permutation_matrix(square_matrix):
#     matrix_p = torch.zeros([*square_matrix.shape])
#     matrix_temp = square_matrix.clone()
#     min_value = square_matrix.min()-1
#     list_i = []
#     list_j = []
#     for r in range(square_matrix.shape[-1]):
#         _,i = torch.max(torch.max(matrix_temp,dim=1)[0], dim=0)
#         _,j = torch.max(torch.max(matrix_temp,dim=0)[0], dim=0)
#         list_i.append(i)
#         list_j.append(j)
#         matrix_temp[i,:] = min_value
#         matrix_temp[:,j] = min_value
#
#     matrix_p[list_i,list_j] = 1.0
#
#     return matrix_p

def create_permutation_matrix(square_matrix_Tensor):
    square_matrix = square_matrix_Tensor.clone().detach().cpu().numpy()
    permutation_matrix = torch.zeros([*square_matrix_Tensor.shape])
    idx, jdx = scipy.optimize.linear_sum_assignment(square_matrix, maximize=True)
    permutation_matrix[idx,jdx] = 1.0

    return permutation_matrix


@khronos
def create_permutation_matrix_v2(square_matrix):
    matrix_p = torch.zeros([*square_matrix.shape])
    matrix_i = torch.ones([*square_matrix.shape])*torch.arange(square_matrix.shape[-1]).unsqueeze(dim=1)
    matrix_j = torch.ones([*square_matrix.shape])*torch.arange(square_matrix.shape[-1]).unsqueeze(dim=0)

    flat_M = torch.flatten(square_matrix.clone())
    flat_i = torch.flatten(matrix_i)
    flat_j = torch.flatten(matrix_j)

    flat_M_sort, indices = torch.sort(flat_M,descending=True)
    flat_i_sort = flat_i[indices]
    flat_j_sort = flat_j[indices]

    L_v = []
    L_i = []
    L_j = []
    for kdx in range(0,square_matrix.shape[-1]):
        max_value = flat_M_sort[0]
        max_i_indices = flat_i_sort[0]
        max_j_indices = flat_j_sort[0]

        L_v.append(max_value.item())
        L_i.append(max_i_indices.item())
        L_j.append(max_j_indices.item())

        # Filter first with i then with j
        out_with_idx = flat_i_sort!=max_i_indices
        flat_M_sort = flat_M_sort[out_with_idx]
        flat_j_sort = flat_j_sort[out_with_idx]
        flat_i_sort = flat_i_sort[out_with_idx]

        out_with_jdx = flat_j_sort!=max_j_indices
        flat_M_sort = flat_M_sort[out_with_jdx]
        flat_j_sort = flat_j_sort[out_with_jdx]
        flat_i_sort = flat_i_sort[out_with_jdx]

    matrix_p[L_i,L_j] = 1.0
    return matrix_p

# @khronos
# def permutation_matrix_v3(Matrix):
#     return permutation_matrix.permutation_matrix(Matrix)


# THE TEST
if __name__ == "__main__":
    Test_Matrix = torch.tensor([[0,0,1,10,1],
                                [2,11,6,7,-2],
                                [3,1,9,0,12],
                                [6,2,4,1,0],
                                [8,9,0,20,13]])

    GT_Permutation_Matrix = torch.tensor([[0,0,1,0,0],
                                          [0,1,0,0,0],
                                          [0,0,0,0,1],
                                          [1,0,0,0,0],
                                          [0,0,0,1,0]])

    estimated_permutation_matrix = create_permutation_matrix(Test_Matrix)
    estimated_permutation_matrix_2 = create_permutation_matrix_v2(Test_Matrix)
    estimated_permutation_matrix_3 = Hungarian_permutation(Test_Matrix)
    estimated_permutation_matrix_4 = scipy_help_for_permutation(Test_Matrix)

    print(torch.eq(GT_Permutation_Matrix,estimated_permutation_matrix).min())
    print(torch.eq(GT_Permutation_Matrix,estimated_permutation_matrix_2).min())
    print(torch.eq(GT_Permutation_Matrix, estimated_permutation_matrix_3).min())
    print(torch.eq(GT_Permutation_Matrix, estimated_permutation_matrix_4).min())

    m =10
    print(f'Square matrix: {m} by {m}')
    Test_Matrix = torch.rand(m,m)
    estimated_permutation_matrix = create_permutation_matrix(Test_Matrix)
    estimated_permutation_matrix = create_permutation_matrix_v2(Test_Matrix)
    estimated_permutation_matrix = Hungarian_permutation(Test_Matrix)
    estimated_permutation_matrix_4 = scipy_help_for_permutation(Test_Matrix)

    m = 100
    print(f'Square matrix: {m} by {m}')
    Test_Matrix = torch.rand(m,m)
    estimated_permutation_matrix = create_permutation_matrix(Test_Matrix)
    estimated_permutation_matrix = create_permutation_matrix_v2(Test_Matrix)
    estimated_permutation_matrix = Hungarian_permutation(Test_Matrix)
    estimated_permutation_matrix_4 = scipy_help_for_permutation(Test_Matrix)

    m = 1000
    print(f'Square matrix: {m} by {m}')
    Test_Matrix = torch.rand(m,m)
    estimated_permutation_matrix = create_permutation_matrix(Test_Matrix)
    estimated_permutation_matrix_4 = scipy_help_for_permutation(Test_Matrix)
    estimated_permutation_matrix = create_permutation_matrix_v2(Test_Matrix)
    #estimated_permutation_matrix = Hungarian_permutation(Test_Matrix)
    estimated_permutation_matrix_4 = scipy_help_for_permutation(Test_Matrix)

    m = 1620
    print(f'Square matrix: {m} by {m}')
    Test_Matrix = torch.rand(m,m)
    estimated_permutation_matrix_4 = scipy_help_for_permutation(Test_Matrix)
    estimated_permutation_matrix = create_permutation_matrix(Test_Matrix)
    # estimated_permutation_matrix = create_permutation_matrix_v2(Test_Matrix)
    # estimated_permutation_matrix = Hungarian_permutation(Test_Matrix)



