# Contains all aspects needed for the short- and long-term memories.

# by Stéphane Vujasinovic

import numpy as np
import torch
import os
import sys
import yaml

from icecream import ic

from HoliMem.ST_HoliMem import ST_HoliMem
from HoliMem.LT_HoliMem import LT_HoliMem
sys.path.append("..")

from Toolbox_Eteph.Debugging.chronometre import khronos_metron


from scipy.signal import tukey


def get_path_2_image_folder(self, img_folder: str):
    self.path_2_image_folder = img_folder


class HoliMem:
    def __init__(self, yaml_HoliMem_config_path:str, debugging_flag=False):
        # debugging
        self.debugging_flag = debugging_flag
        if self.debugging_flag:
            ic.enable()
        else:
            ic.disable()

        # Initialize
        self.init_ST_Memory_flag = True
        self.init_LT_Memory_flag = True

        # Config HoliMem
        ST_config, LT_config = self._read_HoliMem_config(yaml_HoliMem_config_path)

        # Initialize ST and LT modules for HoliMem
        self.ST_HoliMem = ST_HoliMem(ST_config)
        self.LT_HoliMem = LT_HoliMem(LT_config)

        # self.debugging_flag = True


    @property
    def nbr_of_objects_working_with(self):
        return self.nbr_of_objects_in_sequence


    @nbr_of_objects_working_with.setter
    def nbr_of_objects_working_with(self, nbr_of_objects_in_sequence:int):
        self.nbr_of_objects_in_sequence = nbr_of_objects_in_sequence


    @property
    def LT_gram_det(self):
        self.LT_HoliMem._compute_gram_det()
        return self.LT_HoliMem._get_gram_determinant()


    def set_affinity_matrices(self, affinity_matrices:torch.Tensor):
        ic('sub_check1')
        if self.use_affinity_base:
            affinity_matrices = affinity_matrices[1]
            # print(affinity_matrices)
        else:
            affinity_matrices = affinity_matrices[0]
            # print(affinity_matrices)


        ic('sub_check2')
        affinity_matrices_view = affinity_matrices.view(-1,affinity_matrices.shape[-1],affinity_matrices.shape[-1])
        if not self.init_ST_Memory_flag:
            self.mask_st = np.isin(np.array(self.Holistic_frames_indexes_list), np.array(self.ST_frame_indexes_list))

            ic('sub_check3')
            affinity_matrices_ST = affinity_matrices_view[self.mask_st]  # Filter affinity matrix relevant for the short-term memory
            ic('sub_check4')
            self.ST_HoliMem.set_affinity_matrices(affinity_matrices_ST)

        if not self.init_LT_Memory_flag:
            self.mask_lt = np.isin(np.array(self.Holistic_frames_indexes_list), np.array(self.LT_frame_indexes_list))

            ic('sub_check3')
            affinity_matrices_LT = affinity_matrices_view[self.mask_lt]  # Filter affinity matrix relevant for the long-term memory
            ic('sub_check4')
            self.LT_HoliMem.set_affinity_matrices(affinity_matrices_LT)


    # @khronos_metron
    def _read_HoliMem_config(self, yaml_config_file:str):
        with open(yaml_config_file, 'r') as yaml_file:  # Read yaml file and load params
            params = yaml.full_load(yaml_file)
            ST_config = params['ST_Mem']
            LT_config = params['LT_Mem']
            self.ST_LT_modulation = params['ST_LT_modulation']  # TODO, does nothing atm
            self.use_LT_memory_flag = params['Use_LT_memory']
            self.use_affinity_base = params['use_affinity_base']

        return ST_config, LT_config

    # @khronos_metron
    def reset_HoliMem(self):
        self.init_ST_Memory_flag = True
        self.init_LT_Memory_flag = True

        self.ST_HoliMem.reset_Mem()
        self.LT_HoliMem.reset_Mem()


    def update_HoliMem_based_on_crop_mode_3(self, idx, key, value, extracted_feature_for_current_crop):
        # Update the short- and long-term memories
        if self.init_ST_Memory_flag:
            self.init_ST_Memory_flag = False
            self.ST_HoliMem.update_Mem(idx, key, value)
        else:
            self.ST_HoliMem.update_Mem(idx, key, value)

        if not self.use_LT_memory_flag: return  # Skip Long term memory

        if self.init_LT_Memory_flag:
            self.init_LT_Memory_flag = False
            self.LT_HoliMem.update_Mem_with_annotated_frame(idx, key, value)
            self.LT_HoliMem.udpate_crop_Mem_with_annotated_frame(extracted_feature_for_current_crop)
            self.LT_HoliMem.initialize_Gram_matrix_with_features_from_crops_mode_3(idx, key, value, extracted_feature_for_current_crop)
        else:
            # Prepare the short-term memory
            ST_gamma_diversity_ST = self.ST_HoliMem.compute_ST_gamma_diversity()
            # ic(ST_gamma_diversity_ST)

            # Update the long-term memory based on the knowledge of the short-term memory
            # print(extracted_feature_for_current_crop.shape)
            self.LT_HoliMem.update_Mem_crop_mode_3(idx, key, value, extracted_feature_for_current_crop, ST_gamma_diversity_ST)
            # self.LT_HoliMem.construct_the_Gram_matrix(key)




    def find_best_LT_idx(self, idx, key, value, c_key, c_value): # for crop
        # Update the short- and long-term memories
        if self.init_ST_Memory_flag:
            self.init_ST_Memory_flag = False
            self.ST_HoliMem.update_Mem(idx, key, value)
        else:
            self.ST_HoliMem.update_Mem(idx, key, value)

        if not self.use_LT_memory_flag: return  # Skip Long term memory

        if self.init_LT_Memory_flag:
            self.init_LT_Memory_flag = False
            self.LT_HoliMem.update_Mem_with_annotated_frame(idx, key, value)
            self.LT_HoliMem.initialize_Gram_matrix(idx, key, value)
        else:
            # Prepare the short-term memory
            ST_gamma_diversity_ST = self.ST_HoliMem.compute_ST_gamma_diversity()
            # ic(ST_gamma_diversity_ST)

            # Update the long-term memory based on the knowledge of the short-term memory
            self.LT_HoliMem.update_Mem_crop_mode(idx, key, value, c_key, c_value, ST_gamma_diversity_ST)
            # self.LT_HoliMem.construct_the_Gram_matrix(key)

    def update_HoliMem_values_too(self, idx, key, value, strided_center, strided_box):
        # Update the short- and long-term memories
        if self.init_ST_Memory_flag:
            self.init_ST_Memory_flag = False
            self.ST_HoliMem.update_Mem(idx, key, value)
        else:
            self.ST_HoliMem.update_Mem(idx, key, value)

        if not self.use_LT_memory_flag: return  # Skip Long term memory

        if self.init_LT_Memory_flag:
            self.init_LT_Memory_flag = False
            self.LT_HoliMem.update_Mem_with_annotated_frame(idx, key, value)
            self.LT_HoliMem.initialize_Gram_matrix(idx, key, value)
        else:
            # Prepare the short-term memory
            ST_gamma_diversity_ST = self.ST_HoliMem.compute_ST_gamma_diversity()
            # ic(ST_gamma_diversity_ST)

            # Update the long-term memory based on the knowledge of the short-term memory
            self.LT_HoliMem.update_Mem_values_too(idx, key, value, ST_gamma_diversity_ST, strided_center, strided_box)
            # self.LT_HoliMem.construct_the_Gram_matrix(key)


    def update_HoliMem(self, idx, key, value):
        # Update the short- and long-term memories
        if self.init_ST_Memory_flag:
            self.init_ST_Memory_flag = False
            self.ST_HoliMem.update_Mem(idx, key, value)
        else:
            self.ST_HoliMem.update_Mem(idx, key, value)

        if not self.use_LT_memory_flag: return  # Skip Long term memory

        if self.init_LT_Memory_flag:
            self.init_LT_Memory_flag = False
            self.LT_HoliMem.update_Mem_with_annotated_frame(idx, key, value)
            self.LT_HoliMem.initialize_Gram_matrix(idx, key, value)
        else:
            # Prepare the short-term memory
            ST_gamma_diversity_ST = self.ST_HoliMem.compute_ST_gamma_diversity()
            # ic(ST_gamma_diversity_ST)

            # Update the long-term memory based on the knowledge of the short-term memory
            self.LT_HoliMem.update_Mem(idx, key, value, ST_gamma_diversity_ST)
            # self.LT_HoliMem.construct_the_Gram_matrix(key)

    # @khronos_metron
    def get_holistic_memory(self):
        self.Holistic_frames_indexes_list = []

        # Return only the ST memory because u don't care about LT memory
        if not self.use_LT_memory_flag:
            self.ST_frame_indexes_list, _, _ = [*self.ST_HoliMem.read_Mem]
            self.Holistic_frames_indexes_list = self.ST_frame_indexes_list
            return [*self.ST_HoliMem.read_Mem]

        # Combine the ST and LT memories
        self.ST_frame_indexes_list, ST_memory_keys, ST_memory_values = [*self.ST_HoliMem.read_Mem]
        self.LT_frame_indexes_list, LT_memory_keys, LT_memory_values = [*self.LT_HoliMem.read_Mem]

        # print(self.ST_frame_indexes_list)
        # print(self.LT_frame_indexes_list)

        # Filter duplicates and arrange the memory frames in ascending order
        ST_frame_indexes_list_array = np.array(self.ST_frame_indexes_list)
        LT_frame_indexes_list_array = np.array(self.LT_frame_indexes_list)
        _,m = np.unique(LT_frame_indexes_list_array, True)
        LT_frame_indexes_list_array = LT_frame_indexes_list_array[m]
        m = np.isin(ST_frame_indexes_list_array, LT_frame_indexes_list_array, invert = True)        # Need another mask for the LT fames indexes
        sort_w_idx = np.argsort(np.concatenate((LT_frame_indexes_list_array,ST_frame_indexes_list_array[m])),axis=0)

        # Create Holistic memory
        self.Holistic_frames_indexes_list = np.concatenate((LT_frame_indexes_list_array,ST_frame_indexes_list_array[m]),axis=0)[sort_w_idx].tolist()

        # print(self.Holistic_frames_indexes_list)
        #
        #
        # print(self.ST_frame_indexes_list)
        # print(self.LT_frame_indexes_list)

        if [] == self.ST_frame_indexes_list: # When not using ST for the adj frame
            self.Holistic_frames_indexes_list = self.LT_frame_indexes_list
            return self.Holistic_frames_indexes_list, LT_memory_keys, LT_memory_values


        Holistic_memory_keys = torch.concat((LT_memory_keys, ST_memory_keys[:, :, m]), dim=2)[:,:,sort_w_idx]
        Holistic_memory_values = torch.concat((LT_memory_values, ST_memory_values[:, :, m]), dim=2)[:,:,sort_w_idx]

        if self.debugging_flag:
            print(self.ST_frame_indexes_list)
            print(self.LT_frame_indexes_list)
            print(self.Holistic_frames_indexes_list)
            txt = '-------------------------------------------------------------------------------------------------------'
            print(txt)

        return self.Holistic_frames_indexes_list, Holistic_memory_keys, Holistic_memory_values


    @property
    def get_size_of_ST_N_LT_memory(self):
        return self.ST_HoliMem.max_size_of_memory, self.LT_HoliMem.max_size_of_memory

    @property
    def ST_Memory_indexes(self):
        return self.ST_HoliMem.read_Mem[0]

    @ST_Memory_indexes.setter
    def ST_Memory_indexes(self, new_indexes:list):
        self.ST_HoliMem.read_Mem[0] = new_indexes   # TODO: Not ready

    @property
    def LT_Memory_indexes(self):
        return self.LT_HoliMem.read_Mem[0]

    @LT_Memory_indexes.setter
    def LT_Memory_indexes(self, new_indexes:list):
        self.ST_HoliMem.read_Mem[0] = new_indexes   # TODO Not ready

    # @property
    # def are_ST_N_LT_Mems_updated(self):
    #     return self.ST_HoliMem.is_ST_Mem_going_to_be_updated_next_iteration, self.LT_HoliMem.is_LT_Mem_updated


    def is_ST_Mem_going_to_be_updated_next_iteration(self, idx):
        return self.ST_HoliMem.is_ST_Mem_going_to_be_updated_next_iteration(idx)

    @property
    def are_ST_N_LT_Mems_full(self):
        return self.ST_HoliMem.is_ST_Mem_full

    @property
    def access_ST_Memory_elements(self):
        return self.ST_HoliMem.access_ST_Memory_elements


    def re_write_ST_Memory_elements(self, n_ST_list_idx, n_memory_keys, n_memory_values, n_gram_matrix):
        self.ST_HoliMem.re_write_ST_Memory_elements(n_ST_list_idx, n_memory_keys, n_memory_values, n_gram_matrix)













