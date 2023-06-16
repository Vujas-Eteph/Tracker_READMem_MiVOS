# Functions for helping

# by Stéphane Vujasinovic

import numpy as np
import torch
import torch.nn.functional as F

from scipy.signal import tukey

from icecream import ic


def compute_determinant(matrix:np.array):
    return np.linalg.det(matrix)

def compute_similarity_between_two_features(feature_t, features_m, mode=2, device = 'cuda:0'):
    '''
    feature_t --> [1,C,1,H,W] Query
    features_m --> [1,C,N-1,H,W] Memory stack
    mode=0 --> How to compute the similarity
    '''
    # feature_t = torch.unsqueeze(feature_t, 1)
    # ic('****************************')
    # ic(feature_t.shape)
    # ic(features_m.shape)
    # ic('****************************')
    if 0 == mode:
        return F.conv2d(feature_t.permute(1, 0, 2, 3), features_m.permute(1, 0, 2, 3))

    elif 1 == mode:
        cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        A = torch.flatten(feature_t.permute(1, 0, 2, 3), start_dim=1)
        B = torch.flatten(features_m.permute(1, 0, 2, 3), start_dim=1)
        return cos_similarity(A, B)

    elif 2 == mode:
        alpha = 0.5
        cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        kernel_size_y = feature_t.shape[-2]
        kernel_size_x = feature_t.shape[-1]
        win = np.outer(tukey(kernel_size_y, alpha), tukey(kernel_size_x, alpha))

        new_feature_t = feature_t.clone()*torch.Tensor(win).to(device)
        new_features_m = features_m.clone()*torch.Tensor(win).to(device)


        A = torch.flatten(new_feature_t.permute(1, 0, 2, 3), start_dim=1)
        B = torch.flatten(new_features_m.permute(1, 0, 2, 3), start_dim=1)

        return cos_similarity(A, B)


def construct_gram_matrix(memory_elemets, k):
    # self.k # number of objects (self.k, t, 1, nh, nw)

    # ic(self.k)

    for object_id in range(0, k):
        feature_space_in_memory_for_object = memory_elemets[object_id]
        # ic('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        # ic(object_id)
        # ic(memory_elemets.shape)
        # ic(feature_space_in_memory_for_object.shape)
        for idx_memory_frame in range(0, feature_space_in_memory_for_object.shape[1]):
            # ic(idx_memory_frame)
            vector_for_gram = compute_similarity_between_two_features(feature_space_in_memory_for_object[:,idx_memory_frame].unsqueeze(dim=1),
                                                                      feature_space_in_memory_for_object)
            vector_for_gram = vector_for_gram.squeeze().detach().cpu().numpy()
            if 0 != idx_memory_frame:
                gram_matrix_for_object = np.vstack((gram_matrix_for_object, vector_for_gram))
            else:
                gram_matrix_for_object = vector_for_gram

        # ic(gram_matrix_for_object)

        # Use an average if multiple objects
        if k > 1:
            if 0 == object_id:
                avg_gram_matrix = np.zeros([k, *gram_matrix_for_object.shape])
            avg_gram_matrix[object_id] = gram_matrix_for_object
            if k - 1 == object_id:
                gram_matrix = np.average(avg_gram_matrix, axis=0)
        else:
            gram_matrix = gram_matrix_for_object

    # ic(gram_matrix)
    return gram_matrix



def compute_similarity_with_memory_frames_and_current_frame(current_frame, memory_frames, affinity_matrices):
    similarites_list = []
    NBR_OF_OBJECTS, CHANNEL_SIZE = [*memory_frames.shape[:2]]
    SIZE = np.multiply(*memory_frames.shape[-2:])
    query_frame = current_frame[:,:,0]

    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    # There is a better way and faster way to do this. TODO: Optimize after it proves to be efficient by using matrix multplication instead.
    for frame_idx in range(0, memory_frames.shape[2]):
        Matrix_view_of_MEM_frame = memory_frames[:,:,frame_idx].view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE)
        Matrix_view_of_MEM_frame_OG = Matrix_view_of_MEM_frame.clone()
        Matrix_view_of_CUR_frame = query_frame.view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE)

        affinity_matrices = affinity_matrices[frame_idx].clone()
        values, indices = torch.topk(affinity_matrices, k=1, dim=0)  # Extract the best score in the affinity along the query dimension
        sparse_affinity_matrices = torch.ones([*indices.shape]).cuda()  # Set ones where the best score along the query position.
        affinity_matrices.zero_().scatter_(0, indices, sparse_affinity_matrices)
        Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame@affinity_matrices

        similarities = [cos_similarity(Matrix_view_of_MEM_frame[obj_idx].flatten().unsqueeze(axis=0),
                                       Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx in range(0,NBR_OF_OBJECTS)]

        similarities_OG = [cos_similarity(Matrix_view_of_MEM_frame_OG[obj_idx].flatten().unsqueeze(axis=0),
                                       Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                        in range(0, NBR_OF_OBJECTS)]

        similarites_list.append(np.array(similarities).mean())

    return similarites_list


def construct_a_gram_matrix(similarities, frame_idx, gram_matrix_base, max_mem_size, idx_out):
    prev_gram_matrix = gram_matrix_base.copy()
    cardinality_gram_matrix = len(frame_idx)  # +1 for the current frame that is not stored in the memory
    if [] != prev_gram_matrix:
        gram_matrix = np.eye(cardinality_gram_matrix)

        # filter the prev_gram_matrix# Here taking the first element since only working currently on the ST
        if cardinality_gram_matrix == max_mem_size:
            prev_gram_matrix_1 = prev_gram_matrix[:, :idx_out]
            prev_gram_matrix_2 = prev_gram_matrix[:, idx_out + 1:]
            prev_gram_matrix_1_1 = prev_gram_matrix_1[:idx_out, :]
            prev_gram_matrix_2_1 = prev_gram_matrix_1[idx_out + 1:, :]
            prev_gram_matrix_1_2 = prev_gram_matrix_2[:idx_out, :]
            prev_gram_matrix_2_2 = prev_gram_matrix_2[idx_out + 1:, :]
            gram_matrix[:idx_out, :idx_out] = prev_gram_matrix_1_1
            try:
                gram_matrix[idx_out:-1, :idx_out] = prev_gram_matrix_2_1
            except ValueError:
                ic('hi')
                raise 'Error'
            gram_matrix[:idx_out, idx_out:-1] = prev_gram_matrix_1_2
            gram_matrix[idx_out:-1, idx_out:-1] = prev_gram_matrix_2_2
        else:
            gram_matrix[:prev_gram_matrix.shape[0], :prev_gram_matrix.shape[1]] = prev_gram_matrix

    else:
        gram_matrix = np.eye(cardinality_gram_matrix)

    for j in range(cardinality_gram_matrix):
        if cardinality_gram_matrix == j: continue
        gram_matrix[-1, j] = similarities[j]
        gram_matrix[j, -1] = similarities[j]

    return gram_matrix






def construct_gram_matrix_w_affinity_matrices(memory_elemets, k, affinity_matrices):
    size = np.multiply(*memory_elemets.shape[-2:])
    number_of_frames_memory = memory_elemets[0]
    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)



    for object_id in range(0, k):
        feature_space_in_memory_for_object = memory_elemets[object_id]
        feature_space_in_memory_for_object = feature_space_in_memory_for_object.view(-1, k + 1, size)
        # ic('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        # ic(object_id)
        # ic(memory_elemets.shape)
        # ic(feature_space_in_memory_for_object.shape)
        for idx_memory_frame in range(0, feature_space_in_memory_for_object.shape[1]):

            vector_before_affinity = feature_space_in_memory_for_object[:, idx_memory_frame].view(-1, size)
            vector_after_affinity = vector_before_affinity@affinity_matrices[object_id]

            vector_for_gram = compute_similarity_between_two_features(feature_space_in_memory_for_object[:,idx_memory_frame].unsqueeze(dim=1),
                                                                      feature_space_in_memory_for_object)
            vector_for_gram = vector_for_gram.squeeze().detach().cpu().numpy()
            if 0 != idx_memory_frame:
                gram_matrix_for_object = np.vstack((gram_matrix_for_object, vector_for_gram))
            else:
                gram_matrix_for_object = vector_for_gram

        # ic(gram_matrix_for_object)

        # Use an average if multiple objects
        if k > 1:
            if 0 == object_id:
                avg_gram_matrix = np.zeros([k, *gram_matrix_for_object.shape])
            avg_gram_matrix[object_id] = gram_matrix_for_object
            if k - 1 == object_id:
                gram_matrix = np.average(avg_gram_matrix, axis=0)
        else:
            gram_matrix = gram_matrix_for_object

    # ic(gram_matrix)
    return gram_matrix