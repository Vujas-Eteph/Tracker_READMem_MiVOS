# Atomic class for the short- and long-term classes found in ST__HoliMem.py and LT_HoliMem.py

# by Stéphane Vujasinovic

import numpy as np
import torch
import abc
from icecream import ic

# from HoliMem.utils.HoliMem_utils import *

from HoliMem.permutation_matrix_creator import create_permutation_matrix

import lovely_tensors as lt
lt.monkey_patch()


def _compute_similarity_with_memory_frames_and_current_frame_mode_3(features_f16:torch.Tensor, memory_features:torch.Tensor,
                                                                    use_tukey:bool, tukey_alpha:float, debug_flag = False):
    similarites_list = []
    NBR_of_CROPS_IN_MEMORY = memory_features.shape[0]
    # print(memory_features.shape)
    # print(NBR_of_CROPS_IN_MEMORY)

    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    for frame_idx in range(0,NBR_of_CROPS_IN_MEMORY):
        memory_feature_f16 = memory_features[frame_idx]

        # print(features_f16.shape)

        # print(features_f16.flatten().shape)
        Test = cos_similarity(features_f16.flatten().unsqueeze(axis=0), features_f16.flatten().unsqueeze(axis=0)).item() # See if I get 1
        # print(Test)

        similarity = cos_similarity(memory_feature_f16.flatten().unsqueeze(axis=0),features_f16.flatten().unsqueeze(axis=0)).item()

        similarites_list.append(similarity)

    # print(similarites_list)

    return similarites_list


def _compute_similarity_with_memory_frames_and_current_frame(use_cosine_sim, key:torch.Tensor, memory_keys:torch.Tensor,
                                                             affinity_matrices:torch.Tensor,
                                                             use_affinity:bool, use_tukey:bool, tukey_alpha:float, debug_flag = False):
    similarites_list = []
    NBR_OF_OBJECTS, CHANNEL_SIZE = [*memory_keys.shape[:2]]
    SIZE = np.multiply(*memory_keys.shape[-2:])
    query_frame = key[:, :, 0]


    #####################################################################
    # use_cosine_sim = False
    # use_cosine_sim = True
    #######################################################################
    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    dot_poduct = lambda x,y: torch.matmul(x.squeeze(),y.squeeze()) # both elements are flat vectors

    # use_proper_permutation_matrix = True
    use_proper_permutation_matrix = False

    use_tukey = False
    if use_tukey:
        from scipy.signal import tukey
        ALPHA = tukey_alpha
        # print(key.shape)
        kernel_size_y = key.shape[-2]
        kernel_size_x = key.shape[-1]
        # ALPHA SHOULD BE THE INVERSE OF THE SIZE OF THE OBJ INSTANCE //16
        # print(kernel_size_y)
        # print(kernel_size_x)

        win = np.outer(tukey(kernel_size_y, ALPHA), tukey(kernel_size_x, ALPHA))
        # print(win)
        # print('hi')

    Matrix_view_of_CUR_frame = query_frame.view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE).clone()

    for frame_idx in range(0, memory_keys.shape[2]):
        Matrix_view_of_MEM_frame = memory_keys[:, :, frame_idx].view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE).clone()
        # Matrix_view_of_MEM_frame_OG = Matrix_view_of_MEM_frame.clone()

        if use_affinity:
            affinity_matrix = affinity_matrices[frame_idx].clone()
            if not use_proper_permutation_matrix:
                dimension_for_sparse_axis = 0 # For QUERY # yields better results
                # dimension_for_sparse_axis = 1 # For MEMORY #
                # print(affinity_matrix.shape)

                values, indices = torch.topk(affinity_matrix, k=1,
                                             dim=dimension_for_sparse_axis)  # Extract the best score in the affinity along the query dimension

                sparse_affinity_matrices = torch.ones(
                    [*indices.shape]).cuda()  # Set ones where the best score along the query position.

                affinity_matrix = affinity_matrix.zero_().scatter_(dimension_for_sparse_axis, indices, sparse_affinity_matrices)

                # print(affinity_matrix.shape)
                # print(Matrix_view_of_MEM_frame.shape)
                if 0 == dimension_for_sparse_axis:
                    Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix
                elif 1 == dimension_for_sparse_axis:
                    Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix.T
                    # Matrix_view_of_CUR_frame = Matrix_view_of_CUR_frame @ affinity_matrix.T

            else:
                # print(memory_keys.shape[2])
                # print(affinity_matrix.shape)
                # print(frame_idx )
                affinity_matrix = create_permutation_matrix(affinity_matrix).cuda()

                dimension_for_sparse_axis = 0 # For QUERY # yields better results
                if 0 == dimension_for_sparse_axis:
                    Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix
                elif 1 == dimension_for_sparse_axis:
                    Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix.T

            if use_tukey:
                device = 'cuda:0'
                # print('Matrix_view_of_CUR_frame:',Matrix_view_of_CUR_frame)
                # print('Matrix_view_of_MEM_frame',Matrix_view_of_MEM_frame)

                Matrix_view_of_CUR_frame_bis = Matrix_view_of_CUR_frame.clone() * torch.Tensor(win).to(device).view(SIZE)
                Matrix_view_of_MEM_frame_bis = Matrix_view_of_MEM_frame.clone() * torch.Tensor(win).to(device).view(SIZE)
                # print('Matrix_view_of_CUR_frame_bis:',Matrix_view_of_CUR_frame_bis)
                # print('Matrix_view_of_MEM_frame_bis:',Matrix_view_of_MEM_frame_bis)
                #
                # print('hi')

        if use_cosine_sim:
            similarities = [cos_similarity(Matrix_view_of_MEM_frame[obj_idx].flatten().unsqueeze(axis=0),
                                           Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                            in range(0, NBR_OF_OBJECTS)]
        else:
            similarities = [dot_poduct(Matrix_view_of_MEM_frame[obj_idx].flatten().unsqueeze(axis=0),
                                       Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                            in range(0, NBR_OF_OBJECTS)]


        # Testing if rounding the similarity gives me better results
        # print(similarities)
        similarities = np.round(similarities, 5)

        # print(memory_keys.shape)
        # print(similarities)

        # similarities_OG = [cos_similarity(Matrix_view_of_MEM_frame_OG[obj_idx].flatten().unsqueeze(axis=0),
        #                                   Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for
        #                    obj_idx
        #                    in range(0, NBR_OF_OBJECTS)]

        similarites_list.append(np.array(similarities).mean())

    # Similarity with the remaining current frame to add at the end of the gram matrix
    if use_cosine_sim:
        similarities_current = [cos_similarity(Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0),
                                               Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                                in range(0, NBR_OF_OBJECTS)]
    else:
        similarities_current = [dot_poduct(Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0),
                                           Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for
                                obj_idx
                                in range(0, NBR_OF_OBJECTS)]
    # print(similarities_current)
    similarities = np.round(similarities_current, 5)
    similarites_list.append(np.array(similarities).mean())


    # print(similarites_list)

    # print(use_affinity)
    # if debug_flag:
        # print('hi')

    return similarites_list



def _compute_similarity_with_memory_frames_and_current_frame_with_memory_values(strided_center, strided_box, key:torch.Tensor,
                                                                                value:torch.Tensor,
                                                                                memory_keys:torch.Tensor,
                                                                                memory_values:torch.Tensor,
                                                                                affinity_matrices:torch.Tensor,
                                                                                use_affinity:bool, use_tukey:bool, tukey_alpha:float, debug_flag = False):
    #strided_center y_ctr,x_ctr, strided_box [y-min,y-max,x_min,x_max]
    similarites_list = []
    NBR_OF_OBJECTS, CHANNEL_SIZE = [*memory_keys.shape[:2]]
    SIZE = np.multiply(*memory_keys.shape[-2:])
    query_frame = key[:, :, 0].clone()
    print(key.shape)
    print(value.shape)
    print('hi')


    #####################################################################
    # use_cosine_sim = False
    use_cosine_sim = True
    #######################################################################
    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    dot_poduct = lambda x,y: torch.matmul(x.squeeze(),y.squeeze()) # both elements are flat vectors

    Filter = torch.zeros(query_frame.shape)
    print(Filter.shape)
    print(query_frame.shape)
    print(strided_box)
    Filter[:, :, int(strided_box[2]):int(strided_box[3]), int(strided_box[0]):int(strided_box[1])] = 1
    print('hi')
    # Filter strided_box
    print(Filter.shape)
    print(query_frame.shape)

    query_frame = query_frame[:, :, int(strided_box[2]):int(strided_box[3]), int(strided_box[0]):int(strided_box[1])]#query_frame*Filter.cuda()

    SIZE_bis = np.multiply(*query_frame.shape[-2:])

    print(query_frame.shape)
    print(CHANNEL_SIZE)
    print(SIZE_bis)
    print(NBR_OF_OBJECTS)
    Matrix_view_of_CUR_frame = query_frame.reshape(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE_bis).clone()

    print(Matrix_view_of_CUR_frame)

    for frame_idx in range(0, memory_keys.shape[2]):
        Matrix_view_of_MEM_frame = memory_keys[:, :, frame_idx].view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE).clone()
        # Matrix_view_of_MEM_frame_OG = Matrix_view_of_MEM_frame.clone()



        use_affinity=True
        if use_affinity:
            dimension_for_sparse_axis = 0 # For QUERY # yields better results
            # dimension_for_sparse_axis = 1 # For MEMORY #
            affinity_matrix = affinity_matrices[frame_idx].clone()
            print(affinity_matrix.shape)

            values, indices = torch.topk(affinity_matrix, k=1,
                                         dim=dimension_for_sparse_axis)  # Extract the best score in the affinity along the query dimension

            sparse_affinity_matrices = torch.ones(
                [*indices.shape]).cuda()  # Set ones where the best score along the query position.

            affinity_matrix = affinity_matrix.zero_().scatter_(dimension_for_sparse_axis, indices, sparse_affinity_matrices)

            print(affinity_matrix.shape)
            print(Matrix_view_of_MEM_frame.shape)
            if 0 == dimension_for_sparse_axis:
                Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix
            elif 1 == dimension_for_sparse_axis:
                Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix.T

            print(Matrix_view_of_MEM_frame)
            Temporary_mem_key_i = Matrix_view_of_MEM_frame.reshape(NBR_OF_OBJECTS, CHANNEL_SIZE, memory_keys.shape[-2], memory_keys.shape[-1])
            print(Temporary_mem_key_i.shape)
            Temporary_mem_key_i = Temporary_mem_key_i[:, :, int(strided_box[2]):int(strided_box[3]),int(strided_box[0]):int(strided_box[1])]
            Matrix_view_of_MEM_frame = Temporary_mem_key_i.reshape(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE_bis).clone()





            # if use_tukey:
            #     device = 'cuda:0'
            #     # print('Matrix_view_of_CUR_frame:',Matrix_view_of_CUR_frame)
            #     # print('Matrix_view_of_MEM_frame',Matrix_view_of_MEM_frame)
            #
            #     Matrix_view_of_CUR_frame_bis = Matrix_view_of_CUR_frame.clone() * torch.Tensor(win).to(device).view(
            #         SIZE)
            #     Matrix_view_of_MEM_frame_bis = Matrix_view_of_MEM_frame.clone() * torch.Tensor(win).to(device).view(
            #         SIZE)
            #     # print('Matrix_view_of_CUR_frame_bis:',Matrix_view_of_CUR_frame_bis)
                # print('Matrix_view_of_MEM_frame_bis:',Matrix_view_of_MEM_frame_bis)
                #
                # print('hi')

            # # Work with values:
            # # What do I get from the values ?? use the values as a filter
            # print(value)
            # print(memory_values)
            # X = torch.sum(value.clone()*Filter.cuda(),dim=1).detach().cpu()
            # Y = torch.sum(memory_values[:,:,-1].clone().unsqueeze(2),dim=1)
            #
            # print(X)
            # print(Y)
            #
            # import matplotlib.pyplot as plt
            #
            #
            #
            # arr = np.asarray(X.numpy()[0].transpose(1,2,0))
            # plt.imshow(arr, cmap='gray')#, vmin=0, vmax=255)
            # plt.show()
            #
            # # arr = np.asarray(X.detach().cpu().numpy()[1].transpose(1,2,0))
            # # plt.imshow(arr, cmap='gray')#, vmin=0, vmax=255)
            # # plt.show()
            #
            # print('hi')


        if use_cosine_sim:
            similarities = [cos_similarity(Matrix_view_of_MEM_frame[obj_idx].flatten().unsqueeze(axis=0),
                                           Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                            in range(0, NBR_OF_OBJECTS)]
        else:
            similarities = [dot_poduct(Matrix_view_of_MEM_frame[obj_idx].flatten().unsqueeze(axis=0),
                                       Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                            in range(0, NBR_OF_OBJECTS)]




        # Testing if rounding the similarity gives me better results
        print(similarities)
        similarities = np.round(similarities, 5)

        print(memory_keys.shape)
        print(similarities)

        # similarities_OG = [cos_similarity(Matrix_view_of_MEM_frame_OG[obj_idx].flatten().unsqueeze(axis=0),
        #                                   Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for
        #                    obj_idx
        #                    in range(0, NBR_OF_OBJECTS)]

        print(Matrix_view_of_MEM_frame)
        similarites_list.append(np.array(similarities).mean())

    if not use_cosine_sim:
        similarities_current = [dot_poduct(Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0),
                                           Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for
                                obj_idx
                                in range(0, NBR_OF_OBJECTS)]
        print(similarities_current)
        similarities = np.round(similarities_current, 5)
        similarites_list.append(np.array(similarities).mean())


    print(similarites_list)

    print(use_affinity)
    if debug_flag:
        print('hi')

    return similarites_list




def _compute_similarity_with_memory_frames_and_current_frame_other_axis(key:torch.Tensor, memory_keys:torch.Tensor,
                                                             affinity_matrices:torch.Tensor,
                                                             use_affinity:bool, use_tukey:bool, tukey_alpha:float):
    similarites_list = []
    NBR_OF_OBJECTS, CHANNEL_SIZE = [*memory_keys.shape[:2]]
    SIZE = np.multiply(*memory_keys.shape[-2:])
    query_frame = key[:, :, 0]

    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    if use_tukey:
        from scipy.signal import tukey
        ALPHA = tukey_alpha
        kernel_size_y = key.shape[-2]
        kernel_size_x = key.shape[-1]
        # ALPHA SHOULD BE THE INVERSE OF THE SIZE OF THE OBJ INSTANCE //16
        win = np.outer(tukey(kernel_size_y, ALPHA), tukey(kernel_size_x, ALPHA))

    for frame_idx in range(0, memory_keys.shape[2]):
        Matrix_view_of_MEM_frame = memory_keys[:, :, frame_idx].view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE).clone()
        # Matrix_view_of_MEM_frame_OG = Matrix_view_of_MEM_frame.clone()
        Matrix_view_of_CUR_frame = query_frame.view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE).clone()

        if use_affinity:
            affinity_matrix = affinity_matrices[frame_idx].clone()
            print(affinity_matrix.shape)
            values, indices = torch.topk(affinity_matrix, k=1,
                                         # dim=1)  # Extract the best score in the affinity along the query dimension
                                         dim=0)  # Extract the best score in the affinity along the memory dimension
            sparse_affinity_matrices = torch.ones(
                [*indices.shape]).cuda()  # Set ones where the best score along the query position.
            affinity_matrix.zero_().scatter_(0, indices, sparse_affinity_matrices)
            Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix

            if use_tukey:
                device = 'cuda:0'
                Matrix_view_of_CUR_frame = Matrix_view_of_CUR_frame * torch.Tensor(win).to(device).view(SIZE)
                Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame * torch.Tensor(win).to(device).view(SIZE)

        similarities = [cos_similarity(Matrix_view_of_MEM_frame[obj_idx].flatten().unsqueeze(axis=0),
                                       Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                        in range(0, NBR_OF_OBJECTS)]

        # similarities_OG = [cos_similarity(Matrix_view_of_MEM_frame_OG[obj_idx].flatten().unsqueeze(axis=0),
        #                                   Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for
        #                    obj_idx
        #                    in range(0, NBR_OF_OBJECTS)]

        similarites_list.append(np.array(similarities).mean())

    return similarites_list


class Atomic_HoliMem:
    def __init__(self):
        self.reset_Mem()


    def reset_Mem(self):
        self.frame_indexes_list = []
        self.memory_keys   = None
        self.memory_values = None
        self.gram_det = None
        self.gram_matrix = None


    def similarity_annotated_frame(self, use_cosine_sim, memory_key_features_of_ANNOTATED_FRAME):
        h,w = memory_key_features_of_ANNOTATED_FRAME.shape[-2:]
        affinity_matrix = torch.diag(torch.ones(h*w))
        # print(affinity_matrix)
        # print(affinity_matrix.shape)
        return _compute_similarity_with_memory_frames_and_current_frame(use_cosine_sim, memory_key_features_of_ANNOTATED_FRAME, memory_key_features_of_ANNOTATED_FRAME, affinity_matrix,
                                                                        use_affinity=False, use_tukey=False, tukey_alpha = 1.0)


    @staticmethod
    def _compute_det(matrix, denominateur):
        return np.absolute(np.linalg.det(matrix/denominateur))
        # return torch.linalg.det(A)     If using torch for det, the use batch and not hust n vy n matrix, is going to ba faster


    @staticmethod
    def _compute_similarities(use_cosine_sim, key:torch.Tensor, memory_keys:torch.Tensor,
                              affinity_matrices:torch.Tensor,
                              use_affinity_flag=False, use_tukey_win=False, tukey_alpha = 1.0) -> list:
        # print(key.shape)
        # print(affinity_matrices.shape)
        return _compute_similarity_with_memory_frames_and_current_frame(use_cosine_sim, key, memory_keys, affinity_matrices,
                                                                        use_affinity_flag, use_tukey_win, tukey_alpha)

    @staticmethod
    def _compute_similarities_with_memory_values_too(strided_center, strided_box,key:torch.Tensor, value:torch.Tensor, memory_keys:torch.Tensor, memory_values:torch.Tensor,
                              affinity_matrices:torch.Tensor,
                              use_affinity_flag=False, use_tukey_win=False, tukey_alpha = 1.0) -> list:
        # print(key.shape)
        # print(affinity_matrices.shape)
        return _compute_similarity_with_memory_frames_and_current_frame_with_memory_values(strided_center, strided_box, key, value, memory_keys, memory_values, affinity_matrices,
                                                                        use_affinity_flag, use_tukey_win, tukey_alpha)


    @staticmethod
    def _compute_similarities_mode_3(features:torch.Tensor, memory_keys:torch.Tensor,
                                     use_tukey_win=False, tukey_alpha = 1.0) -> list:
        return _compute_similarity_with_memory_frames_and_current_frame_mode_3(features, memory_keys, use_tukey_win, tukey_alpha)


    # @staticmethod #  cosine similarity
    # def _update_gram_matrix(gram_matrix:np.array, similarities:list) -> np.array: # version_for_cosine_similarity
    #     print(similarities)
    #     print(gram_matrix)
    #     gram_matrix = np.concatenate((gram_matrix, np.array([similarities]).T), axis=1)
    #     gram_matrix = np.concatenate((gram_matrix, np.array([similarities + [1]])), axis=0)
    #     print(gram_matrix)
    #     return gram_matrix


    # @staticmethod   # Not cosine similarity
    # def _update_gram_matrix_dot(gram_matrix:np.array, similarities:list) -> np.array: # version_for_dot_product # Meilleur version et devrait marcher avec cosine similarity si dans cosine similarity je fait aussi le calcule de la similarité avec le current frame
    #     print(similarities)
    #     print(gram_matrix)
    #     gram_matrix = np.concatenate((gram_matrix, np.array([similarities[:-1]]).T), axis=1)
    #     gram_matrix = np.concatenate((gram_matrix, np.array([similarities])), axis=0)
    #     print(gram_matrix)
    #     return gram_matrix

    @staticmethod
    def _update_gram_matrix(gram_matrix:np.array, similarities:list) -> np.array: # version_for_dot_product # Meilleur version et devrait marcher avec cosine similarity si dans cosine similarity je fait aussi le calcule de la similarité avec le current frame
        # print(similarities)
        # print(gram_matrix)
        gram_matrix = np.concatenate((gram_matrix, np.array([similarities[:-1]]).T), axis=1)
        gram_matrix = np.concatenate((gram_matrix, np.array([similarities])), axis=0)
        # print(gram_matrix)
        return gram_matrix


    @staticmethod
    def _delete_col_N_row_from_Gram_matrix(gram_matrix:np.array, idx_out: int) -> np.array:
        gram_matrix = np.delete(gram_matrix, idx_out, 0)
        gram_matrix = np.delete(gram_matrix, idx_out, 1)
        return gram_matrix


    @abc.abstractmethod
    def update_Mem(self):
        '''
        update the memory
        '''
        raise NotImplementedError("Must be implemented in subclass.")


    @property
    def read_Mem(self):
        '''
        return/read/get memory
        '''
        return self.frame_indexes_list, self.memory_keys, self.memory_values


    def set_affinity_matrices(self, affinity_matrices):
        self.affinity_matrices = affinity_matrices


    def _compute_gram_det(self):
        denominateur = self.gram_matrix[0,0]
        self.gram_det = self._compute_det(self.gram_matrix,denominateur)


    def _get_gram_determinant(self) -> np.array:
        # print('self.gram_det')
        # print(self.gram_det)
        return self.gram_det

