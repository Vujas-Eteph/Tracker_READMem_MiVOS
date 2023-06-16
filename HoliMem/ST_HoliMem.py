# Handles only the short-term memory aspect of HoliMem

# by Stéphane Vujasinovic

import numpy as np
import torch

from HoliMem.utils.HoliMem_utils import *
from HoliMem.HoliMem_Atomic import Atomic_HoliMem

import sys
sys.path.append("..")

from Toolbox_Eteph.Debugging.chronometre import khronos_metron


class ST_HoliMem(Atomic_HoliMem):
    def __init__(self, ST_config):
        Atomic_HoliMem.__init__(self)
        # Configure short-term memory
        self.max_size_of_memory = ST_config['size']
        # print(self.max_size_of_memory)
        self.mem_freq = ST_config['freq']
        self.get_adjacent_frame_into_the_ST_Mem_flag = ST_config['Get_adjacent_frame_into_the_mix_flag']
        self.use_cosine_sim = False # ST_config['use_cosine_sim']
        # self.thrown_keys, self.thrown_values, self.thrown_index, self.thrown_affinity = None, None, None, None

        # self.Use_affinity = ST_config['Use_affinity']
        # self.Use_tukey_window = ST_config['Use_tukey_window']
        # if self.Use_tukey_window:
        #     self.tukey_alpha = ST_config['tukey_alpha']
        # else:
        #     self.tukey_alpha = False

        self.Use_affinity = False
        self.Use_tukey_window = False
        self.tukey_alpha = False
        # if self.Use_tukey_window:
        #     self.tukey_alpha = ST_config['tukey_alpha']
        # else:
        #     self.tukey_alpha = False


    # @khronos_metron
    def update_Mem(self, idx:int, key:torch.Tensor, value:torch.Tensor):
        # Check the gram matrix
        if self.gram_matrix is None:
            self.gram_matrix = np.array([[self._compute_similarities(self.use_cosine_sim, key, key, None)[0]]])

        if self.max_size_of_memory == 0: return
        self.updated_ST_Mem_flag = False
        # The ST Memory follows the FIFO format. At the same time update the gram matrix
        if None is self.memory_keys: # Initialize the ST memory
            self.memory_keys = key
            self.memory_values = value
            self.frame_indexes_list.append(idx)
            self.frame_index_wo_adj_frame = self.frame_indexes_list.copy()  # Mostly for QDMN version
            self.adjacent_frame_in_memory_flag = False  # Now when to discard the previous frame
            self._filter_gram_matrix = False
            self.updated_ST_Mem_flag = True
            return

        # compute similarities, the gram matrix
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # print(self.gram_matrix.shape)
        # print(self.affinity_matrices.shape)
        # print(self._compute_similarities(key, self.memory_keys, self.affinity_matrices))
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # print('max_memory_size:',self.max_size_of_memory)
        self.gram_matrix = self._update_gram_matrix(self.gram_matrix, self._compute_similarities(self.use_cosine_sim, key, self.memory_keys, None, self.Use_affinity, self.Use_tukey_window, self.tukey_alpha))


        # Throw away the adjacent frame store in the previous iteration before updating the memory with a new frame
        if self.adjacent_frame_in_memory_flag:
            self.memory_keys = self.memory_keys[:,:,:-1,:,:]
            self.memory_values = self.memory_values[:,:,:-1,:,:]
            del self.frame_indexes_list[-1]
            self.adjacent_frame_in_memory_flag = False
            self._filter_gram_matrix = True

            self.gram_matrix = self._delete_col_N_row_from_Gram_matrix(self.gram_matrix, idx_out=-2)


        # Ensure that only every n-th frame is taken into the memory
        if idx - self.frame_indexes_list[-1] >= self.mem_freq:
            self.memory_keys = torch.cat([self.memory_keys, key], dim=2)
            self.memory_values = torch.cat([self.memory_values, value], dim=2)
            self.frame_indexes_list.append(idx)
            self.frame_index_wo_adj_frame = self.frame_indexes_list.copy()  # Mostly for QDMN version
            self.updated_ST_Mem_flag = True
        else:
            if not self.get_adjacent_frame_into_the_ST_Mem_flag:
                self.gram_matrix = self._delete_col_N_row_from_Gram_matrix(self.gram_matrix, idx_out=-1)

        # Trim ST Mem
        self.thrown_keys, self.thrown_values, self.thrown_index, self.thrown_affinity = None, None, None, None
        if self.max_size_of_memory < self.memory_keys.shape[2]:
            # self.thrown_keys = self.memory_keys[:, :, :-self.max_size_of_memory].clone()
            # self.thrown_values = self.memory_values[:, :, :-self.max_size_of_memory].clone()
            # self.thrown_index = self.frame_indexes_list[:-self.max_size_of_memory]
            # self.thrown_affinity = self.affinity_matrices[:-self.max_size_of_memory,:,:]

            self.memory_keys = self.memory_keys[:, :, -self.max_size_of_memory:]
            self.memory_values = self.memory_values[:, :, -self.max_size_of_memory:]
            self.frame_indexes_list = self.frame_indexes_list[-self.max_size_of_memory:]

            self.gram_matrix = self._delete_col_N_row_from_Gram_matrix(self.gram_matrix, idx_out=0)


        # Take the adjacent frame into the ST memory
        if not self.get_adjacent_frame_into_the_ST_Mem_flag: return
        if self.frame_indexes_list[-1] != idx:
            self.memory_keys = torch.cat([self.memory_keys, key], dim=2)
            self.memory_values = torch.cat([self.memory_values, value], dim=2)
            self.frame_indexes_list.append(idx)
            self.adjacent_frame_in_memory_flag = True



    # @khronos_metron
    def compute_ST_gamma_diversity(self):
        # The diversity is only used when the long-term memory is. Until then the short-term memory should be fulled.
        if 1 == self.max_size_of_memory or 0 == self.max_size_of_memory: return 0.0
        sum_upper_gram_matrix = np.triu(self.gram_matrix[-self.max_size_of_memory:,-self.max_size_of_memory:], 1).sum()
        # sum_upper_gram_matrix = np.triu(self.gram_matrix[:self.max_size_of_memory,:self.max_size_of_memory], 1).sum()
        # self.ST_Gamma_diversity = 1 - (2 / (self.max_size_of_memory * (self.max_size_of_memory - 1))) * \
        #                           (sum_upper_gram_matrix / self.gram_matrix[:self.max_size_of_memory,:self.max_size_of_memory].max())
        self.ST_Gamma_diversity = 1 - (2 / (self.max_size_of_memory * (self.max_size_of_memory - 1))) * sum_upper_gram_matrix
        return self.get_ST_gamme_diversity()

    # @khronos_metron
    def get_ST_gamme_diversity(self):
        return self.ST_Gamma_diversity



    def is_ST_Mem_going_to_be_updated_next_iteration(self, idx):
        return idx - self.frame_index_wo_adj_frame[-1] >= self.mem_freq# -1

    @property
    def is_ST_Mem_full(self):
        return len(self.frame_index_wo_adj_frame) >= self.max_size_of_memory


    @property
    def access_ST_Memory_elements(self):
        return self.frame_indexes_list.copy(), self.memory_keys.clone(), self.memory_values.clone(), self.gram_matrix.copy()

    def re_write_ST_Memory_elements(self, n_frame_indexes_list, n_memory_keys, n_memory_values, n_gram_matrix):
        self.frame_indexes_list = n_frame_indexes_list
        self.memory_keys = n_memory_keys
        self.memory_values = n_memory_values
        self.gram_matrix = n_gram_matrix


    # def return_thrown_elements(self):
    #     return self.thrown_keys, self.thrown_values, self.thrown_index, self.thrown_affinity





