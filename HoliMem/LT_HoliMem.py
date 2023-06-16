# Handles only the long-term memory aspect of HoliMem

# by Stéphane Vujasinovic

import numpy as np
import torch
from icecream import ic

from HoliMem.utils.HoliMem_utils import *
from HoliMem.HoliMem_Atomic import Atomic_HoliMem

class LT_HoliMem(Atomic_HoliMem):
    def __init__(self, LT_config):
        Atomic_HoliMem.__init__(self)
        # Configure long-term memory
        self.max_size_of_memory = LT_config['size']
        self.mem_freq = LT_config['freq']
        self.similarity_bound = LT_config['similarity_bound']
        self.method_LT_memory = LT_config['method']
        if 'annotated and last' == self.method_LT_memory or 'annotated and max' == self.method_LT_memory:
            self.alpha_A_L_A_M = LT_config['alpha_A_L_A_M']
            print('\nself.alpha_A_L_A_M', self.alpha_A_L_A_M)

        self.keep_annotated_frame_in_LT_memory = LT_config['keep_annotated_frame_in_LT_memory']
        self.Use_affinity = LT_config['Use_affinity']
        self.Use_tukey_window = LT_config['Use_tukey_window']
        if self.Use_tukey_window:
            self.tukey_alpha = LT_config['tukey_alpha']
        else:
            self.tukey_alpha = False

        self.updated_LT_Mem_flag = False
        # LT_mem_config['score_threshold']

        try:
            self.LT_init_method = LT_config['init_method']
        except:
            self.LT_init_method = 'Every_M'
        # self.LT_init_method = 'Full_annotated'  # Basic initialization
        # self.LT_init_method = 'Every_M'       # Initialization very every m frame is used to fill the LT memory

        # self.normalizing_factor_gram_matrix = None

        self.use_cosine_sim = LT_config['use_cosine_sim']



    def set_init_method_for_gram(self, LT_init_method):
        # self.LT_init_method = 'Full_annotated' # possible methods
        #self.LT_init_method = 'Every_M'
        self.LT_init_method = LT_init_method

    # def reset_LT_gram_matrix(self, memory_key_features_of_ANNOTATED_FRAME): # Could also be used for cosine similarity actually
    #     similarities = self.similarity_annotated_frame(self.use_cosine_sim, memory_key_features_of_ANNOTATED_FRAME)
    #     # print(similarities)
    #     # print(self.gram_matrix)
    #     similarity = np.array(similarities).mean()
    #     similarity = np.round(similarity, 5)
    #     # print(similarity)
    #     self.gram_matrix = np.array([[similarity]])
    #     self.normalizing_factor_gram_matrix = similarity
    #     # print(self.gram_matrix)
    #     # print(self.gram_matrix.shape)
    #     # print('Hi')



    def update_Mem_with_annotated_frame(self, idx, key, value):
        # Annotated frame is always kept in the LT memory as it is the one with a guaranteed good mask
        self.frame_indexes_list.append(idx)
        self.memory_keys = key
        self.memory_values = value
        self.affinity_matrices = torch.Tensor(np.eye(key.shape[-2]*key.shape[-1])).to(device='cuda:0').unsqueeze(dim=0)
        self.updated_LT_Mem_flag = True


    def udpate_crop_Mem_with_annotated_frame(self, features_from_crop):
        self.memory_f16 = features_from_crop


    def initialize_Gram_matrix(self, idx, key, value):
        if self.gram_matrix is None:
            self.gram_matrix = np.array([[self._compute_similarities(self.use_cosine_sim, key, key, None)[0]]])
        if 'Every_M' == self.LT_init_method: return
        # Initialize the gram matrix
        self.Current_memory_size = self.memory_keys.shape[2]
        while self.Current_memory_size < self.max_size_of_memory:
            self.gram_matrix = self._update_gram_matrix(self.gram_matrix, self._compute_similarities(self.use_cosine_sim, key, self.memory_keys, self.affinity_matrices,
                                                                                                     self.Use_affinity, self.Use_tukey_window, self.tukey_alpha))

            self.memory_keys = torch.cat([self.memory_keys,key], dim=2)
            self.memory_values = torch.cat([self.memory_values,value], dim=2)
            self.frame_indexes_list.append(idx)

            self.affinity_matrices = torch.cat([self.affinity_matrices,
                                                torch.Tensor(np.eye(key.shape[-2]*key.shape[-1])).to(device='cuda:0').unsqueeze(dim=0)],
                                               dim=0)
            self.Current_memory_size = self.memory_keys.shape[2]


    def initialize_Gram_matrix_with_features_from_crops_mode_3(self, idx, key, value, features_f16): # mode 3
        if 'Every_M' == self.LT_init_method: return
        # Initialize the gram matrix
        self.Current_memory_size = self.memory_keys.shape[2]
        while self.Current_memory_size < self.max_size_of_memory:
            self.gram_matrix = self._update_gram_matrix(self.gram_matrix,
                                                        self._compute_similarities(self.use_cosine_sim,
                                                                                   key,
                                                                                   self.memory_keys,
                                                                                   affinity_matrices = None,
                                                                                   use_affinity_flag=self.Use_affinity,
                                                                                   use_tukey_win=self.Use_tukey_window,
                                                                                   tukey_alpha=self.tukey_alpha))

            self.memory_keys = torch.cat([self.memory_keys,key], dim=2)
            self.memory_values = torch.cat([self.memory_values,value], dim=2)
            self.memory_f16 = torch.cat([self.memory_f16,features_f16], dim=0)
            self.frame_indexes_list.append(idx)

            self.affinity_matrices = torch.cat([self.affinity_matrices,
                                                torch.Tensor(np.eye(key.shape[-2]*key.shape[-1])).to(device='cuda:0').unsqueeze(dim=0)],
                                               dim=0)
            self.Current_memory_size = self.memory_keys.shape[2]



    def update_Mem(self, idx, key, value, ST_gamma_diversity_ST):
        # print(self.gram_matrix/self.gram_matrix[0,0])
        self.updated_LT_Mem_flag = False
        # After condition 0 and 1 are valid, add the new frame in the memory until full.
        # If the LT memroy is at full capacity, then check if the diversity is enhanced by replacing one of the memory frames with the current frame
        _idx, _key, _value = idx, key, value
        if not self._condition_only_every_i_th_frame_to_consider(_idx): return
        print('_idx:',_idx)

        # if 'Every_M' == self.LT_init_method:
        #     if not self._check_that_LT_memory_is_full(idx,key,value): return


        # print(self._condition_for_only_considering_similar_frames(_key, ST_gamma_diversity_ST))
        if not self._condition_for_only_considering_similar_frames(_key, ST_gamma_diversity_ST): return

        self._update_LT_Mem_part_1(_idx, _key, _value)


    # def update_Mem_values_too(self, idx, key, value, ST_gamma_diversity_ST, strided_center, strided_box):
    #     self.updated_LT_Mem_flag = False
    #     # After condition 0 and 1 are valid, add the new frame in the memory until full.
    #     # If the LT memroy is at full capacity, then check if the diversity is enhanced by replacing one of the memory frames with the current frame
    #     _idx, _key, _value = idx, key, value
    #     if not self._condition_only_every_i_th_frame_to_consider(_idx): return
    #     # print(self._condition_for_only_considering_similar_frames(_key, ST_gamma_diversity_ST))
    #     if not self._condition_for_only_considering_similar_frames_values_too(_key, _value, strided_center, strided_box, ST_gamma_diversity_ST): return
    #
    #     self._update_LT_Mem_part_1_values_too(_idx, _key, _value, strided_center, strided_box)


    # def update_Mem_crop_mode_3(self, idx, key, value, features_f16, ST_gamma_diversity_ST): # this is mode 2
    #     self.updated_LT_Mem_flag = False
    #     # After condition 0 and 1 are valid, add the new frame in the memory until full.
    #     # If the LT memroy is at full capacity, then check if the diversity is enhanced by replacing one of the memory frames with the current frame
    #     _idx, _key, _value, _features_f16 = idx, key, value, features_f16
    #     if not self._condition_only_every_i_th_frame_to_consider(_idx): return
    #     # print(self._condition_for_only_considering_similar_frames(_key, ST_gamma_diversity_ST))
    #     if not self._condition_for_only_considering_similar_frames_features_f16(features_f16, ST_gamma_diversity_ST): return
    #
    #     self._update_LT_Mem_part_1_crop_mode_3(_idx, _key, _value, _features_f16)



    # def update_Mem_crop_mode(self, idx, key, value, c_key, c_value, ST_gamma_diversity_ST): # this is mode 2
    #     self.updated_LT_Mem_flag = False
    #     # After condition 0 and 1 are valid, add the new frame in the memory until full.
    #     # If the LT memroy is at full capacity, then check if the diversity is enhanced by replacing one of the memory frames with the current frame
    #     _idx, _key, _value, _c_key, _c_value = idx, key, value, c_key, c_value
    #     if not self._condition_only_every_i_th_frame_to_consider(_idx): return
    #     # print(self._condition_for_only_considering_similar_frames(_key, ST_gamma_diversity_ST))
    #     if not self._condition_for_only_considering_similar_frames(_c_key, ST_gamma_diversity_ST): return
    #
    #     self._update_LT_Mem_part_1_crop_mode(_idx, _key, _value, _c_key, _c_value)


    def _condition_only_every_i_th_frame_to_consider(self, idx):
        # Condition that the new frame is not a redundant information
        # return self.mem_freq <= idx - self.frame_indexes_list[-1] # dynmaic sampling interval
        return 0 == (idx % self.mem_freq) # static sampling intervalrate
        # return (self.mem_freq <= idx - self.frame_indexes_list[-1]) or (0 == (idx % self.mem_freq)) # annealing sampling interval

        # SECOND FORMAT for annealing
        # if 0 == (idx % self.mem_freq): return True
        # if idx//self.mem_freq != self.frame_indexes_list[-1]//self.mem_freq: return True
        # return False


        # return not (self.mem_freq == idx - self.frame_indexes_list[-1])

    # def _init_LT_gram_matrix(self):
    #     self.gram_matrix = np.ones([self.max_size_of_memory,self.max_size_of_memory])  # Init only with ones
    #     annotated_key = self.memory_keys.clone()
    #     annotated_value = self.memory_values.clone()
    #     self.affinity_with_annotated = self.affinity_matrices.clone()
    #
    #     Current_memory_size = self.memory_keys.shape[2]
    #
    #     while Current_memory_size < self.max_size_of_memory:
    #         self.memory_keys = torch.cat([self.memory_keys,annotated_key], dim=2)
    #         self.memory_values = torch.cat([self.memory_values,annotated_value], dim=2)
    #         Current_memory_size = self.memory_keys.shape[2]
    #
    #         self.affinity_matrices = torch.cat([self.affinity_matrices,self.affinity_with_annotated], dim=0)
    #         self.frame_indexes_list.append(0)


    def _update_LT_Mem_part_1(self, idx, key, value):
        if 'Every_M' == self.LT_init_method:
            if not self._check_that_LT_memory_is_full(idx,key,value): return
        # Throw away the annotated frame garanteed
        # if self.discard_annotated_frame(): return
        condition_valid, best_idx = self._condition_to_update_LT_Mem_only_if_diversity_is_enhanced(key)
        if not condition_valid: return
        self._update_LT_Mem_part_2(idx, best_idx, key, value)
        self.updated_LT_Mem_flag = True

    # def _update_LT_Mem_part_1_values_too(self, idx, key, value, strided_center, strided_box):
    #     if 'Every_M' == self.LT_init_method:
    #         if not self._check_that_LT_memory_is_full_values_too(idx,key,value, strided_center, strided_box): return
    #     condition_valid, best_idx = self._condition_to_update_LT_Mem_only_if_diversity_is_enhanced_values_too(key, value, strided_center, strided_box)
    #     if not condition_valid: return
    #     self._update_LT_Mem_part_2(idx, best_idx, key, value)
    #     self.updated_LT_Mem_flag = True

    # def _update_LT_Mem_part_1_crop_mode(self, idx, key, value, _c_key, _c_value): # mode 2
    #     if 'Every_M' == self.LT_init_method:
    #         if not self._check_that_LT_memory_is_full(idx,key,value): return
    #     condition_valid, best_idx = self._condition_to_update_LT_Mem_only_if_diversity_is_enhanced(_c_key)
    #     if not condition_valid: return
    #     self._update_LT_Mem_part_2(idx, best_idx, key, value)
    #     self.updated_LT_Mem_flag = True

    # def _update_LT_Mem_part_1_crop_mode_3(self, idx, key, value, _features_f16): # mode 3
    #     if 'Every_M' == self.LT_init_method:
    #         if not self._check_that_LT_memory_is_full_crop_mode_3(idx,key,value, _features_f16): return
    #     condition_valid, best_idx = self._condition_to_update_LT_Mem_only_if_diversity_is_enhanced_crop_mode_3(_features_f16)
    #     if not condition_valid: return
    #     self._update_LT_Mem_part_2(idx, best_idx, key, value)
    #     self.updated_LT_Mem_flag = True
    #     print(self.memory_f16)
    #     print(_features_f16)
    #
    #     self.memory_f16 = torch.cat([self.memory_f16[:best_idx], self.memory_f16[best_idx + 1:], _features_f16], dim=0)

    def _check_that_LT_memory_is_full(self, idx, key, value):
        # Ensure that the LT memory is full, if not full, then full it.
        Current_memory_size = self.memory_keys.shape[2]
        if Current_memory_size >= self.max_size_of_memory: return True  # if Memory is not complete then

        # Update the gram matrix
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # print(self.gram_matrix.shape)
        # print(self.affinity_matrices.shape)
        # print(self._compute_similarities(key, self.memory_keys, self.affinity_matrices))
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # if self.use_cosine_sim:
        self.gram_matrix = self._update_gram_matrix(self.gram_matrix, self._compute_similarities(self.use_cosine_sim, key, self.memory_keys, self.affinity_matrices))
        # else:
        #     self.gram_matrix = self._update_gram_matrix_dot(self.gram_matrix,
        #                                                 self._compute_similarities(self.use_cosine_sim, key,
        #                                                                            self.memory_keys,
        #                                                                            self.affinity_matrices))

        # Update LT Memory
        self.frame_indexes_list.append(idx)
        self.memory_keys = torch.cat([self.memory_keys, key], dim=2)
        self.memory_values = torch.cat([self.memory_values, value], dim=2)
        return False

    def discard_annotated_frame(self):
        if 0 not in self.frame_indexes_list: return False
        self.frame_indexes_list = self.frame_indexes_list[1:]
        self.memory_keys = self.memory_keys[:,:,1:]
        self.memory_values =self.memory_values[:,:,1:]
        self.gram_matrix = self.gram_matrix[1:,1:]

        return True

    def _check_that_LT_memory_is_full_values_too(self, idx, key, value, strided_center, strided_box):
        # Ensure that the LT memory is full, if not full, then full it.
        Current_memory_size = self.memory_keys.shape[2]
        if Current_memory_size >= self.max_size_of_memory: return True  # if Memory is not complete then

        # Update the gram matrix
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # print(self.gram_matrix.shape)
        # print(self.affinity_matrices.shape)
        # print(self._compute_similarities(key, self.memory_keys, self.affinity_matrices))
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        self.gram_matrix = self._update_gram_matrix(self.gram_matrix, self._compute_similarities_with_memory_values_too(strided_center, strided_box, key, value, self.memory_keys, self.memory_values, self.affinity_matrices))

        # Update LT Memory
        self.frame_indexes_list.append(idx)
        self.memory_keys = torch.cat([self.memory_keys, key], dim=2)
        self.memory_values = torch.cat([self.memory_values, value], dim=2)
        return False

    def _check_that_LT_memory_is_full_crop_mode_3(self, idx, key, value, _features_f16):
        # Ensure that the LT memory is full, if not full, then full it.
        Current_memory_size = self.memory_keys.shape[2]
        if Current_memory_size >= self.max_size_of_memory: return True  # if Memory is not complete then

        # Update the gram matrix
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        # print(self.gram_matrix.shape)
        # print(self.affinity_matrices.shape)
        # print(self._compute_similarities(key, self.memory_keys, self.affinity_matrices))
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        self.gram_matrix = self._update_gram_matrix(self.gram_matrix, self._compute_similarities(self.use_cosine_sim,
                                                                                                 key,
                                                                                                 self.memory_keys,
                                                                                                 affinity_matrices = None,
                                                                                                 use_affinity_flag=self.Use_affinity,
                                                                                                 use_tukey_win=self.Use_tukey_window,
                                                                                                 tukey_alpha = self.tukey_alpha))

        # Update LT Memory
        self.frame_indexes_list.append(idx)
        self.memory_keys = torch.cat([self.memory_keys, key], dim=2)
        self.memory_values = torch.cat([self.memory_values, value], dim=2)
        self.memory_f16 = torch.cat([self.memory_f16, _features_f16], dim=0)
        return False


    def _update_LT_Mem_part_2(self, idx, best_idx, key, value):
        # Replace the element in the memory with the better version
        del self.frame_indexes_list[best_idx]
        self.frame_indexes_list.append(idx)
        self.memory_keys = torch.cat([self.memory_keys[:, :, :best_idx],
                                      self.memory_keys[:, :, best_idx + 1:],
                                      key], dim=2)
        self.memory_values = torch.cat([self.memory_values[:, :, :best_idx],
                                        self.memory_values[:, :, best_idx + 1:],
                                        value], dim=2)



    def _condition_for_only_considering_similar_frames_features_f16(self, features_f16, ST_gamma_diversity = 0):
        # print(features_f16.shape)
        similarities = self._compute_similarities_mode_3(features_f16, self.memory_f16, self.Use_tukey_window, self.tukey_alpha)

        if 'static lower bound' == self.method_LT_memory:
            return similarities[0] > self.similarity_bound
        elif 'dynamic lower bound' == self.method_LT_memory:
            return similarities[0] > self.similarity_bound - ST_gamma_diversity
        elif 'ensemble lower bound' == self.method_LT_memory:
            return np.array(similarities).mean() > self.similarity_bound
            # not all(curr_sims_norm > self._lb)
            # print(self._lt_gram_matrix)
        #     ensemble_similarities = [compute_similarity_between_two_features(key[obj_idx], self.memory_keys[
        #         obj_idx]).cpu().detach().numpy() for obj_idx in range(0, nbr_of_objects)]
        #     # elif "values" == use_keys_or_values:
        #     #     ensemble_similarities = [compute_similarity_between_two_features(prev_value[obj_idx],self.LT_memory_values[obj_idx]).cpu().detach().numpy()  for obj_idx in range(0, self.k)]
        #     if self.gram_matrix.shape != ():
        #         return (
        #                     np.array(ensemble_similarities).mean(axis=0).squeeze() > self.similarity_bound * np.diag(
        #                 self.gram_matrix)).min()
        #     else:
        #         return np.array(ensemble_similarities).mean(
        #             axis=0).squeeze() > self.similarity_bound * self.gram_matrix.shape



    def _condition_for_only_considering_similar_frames(self, key, ST_gamma_diversity = 0):
        # Check with the similarity of the current frame and the ones in the LT memory.
        # If similarity too low, might be because too much background taken into account.
        values,indexes,counts = np.unique(np.array(self.frame_indexes_list),True,False,True)

        # print('values,indexes,counts')
        # print(values,indexes,counts)

        # Just the clone, copy the affinity of the annotated frame and give it to the other also! ATTENTION ! ONLY FOR THE INITIALIZATION !!
        for v, i, c in zip(values, indexes,counts):  # This only works at the initialization and is not a general solution. !!!!!
            if 1 == c: continue

            affinity_to_clone = self.affinity_matrices[i,:,:].clone() # Take the index 0 because its the annotated variant
            new_affinities = torch.zeros(c-1,*affinity_to_clone.shape).cuda()
            for j in range(0,c-1):
                new_affinities[j]+=affinity_to_clone

            self.affinity_matrices = torch.cat([new_affinities, self.affinity_matrices], dim=0) # Add at the beginning of the extract affinity matrices, as you want to add affiniies only for the annotation frame

        # Create temporary_affinity_matrix that matches the number of elements in the memory. Very important to be constistent
        # with the number of frames stored in the lt memory and the affinity matrices.

        similarities = self._compute_similarities(self.use_cosine_sim, key, self.memory_keys, self.affinity_matrices, self.Use_affinity, self.Use_tukey_window, self.tukey_alpha)
        similarity_with_annotated_frame = similarities[0]

        # See if second condition is valid: That the similarity is good enough
        # print(similarity_with_annotated_frame)
        # print(self.similarity_bound)
        if 'static lower bound' == self.method_LT_memory:
            return similarity_with_annotated_frame/self.gram_matrix[0,0]> self.similarity_bound

        elif 'dynamic lower bound' == self.method_LT_memory:
            return similarity_with_annotated_frame/self.gram_matrix[0,0] > self.similarity_bound - ST_gamma_diversity

        elif 'ensemble lower bound' == self.method_LT_memory:
            # print(similarities)
            return np.array(similarities).mean()/self.gram_matrix[0,0] > self.similarity_bound

        elif 'last' == self.method_LT_memory:
            # print(similarities)
            return similarities[-1]/self.gram_matrix[0,0] > self.similarity_bound

        elif 'ensemble max' == self.method_LT_memory:
            # print(similarities)
            return np.array(similarities[:-1]).max()/self.gram_matrix[0,0] > self.similarity_bound

        elif 'annotated and last' == self.method_LT_memory:
            ALPHA = self.alpha_A_L_A_M
            print(ALPHA)
            print(similarities)
            print(((1-ALPHA)*similarities[-1] + ALPHA*similarity_with_annotated_frame)/self.gram_matrix[0, 0])
            return ((1-ALPHA)*similarities[-1] + ALPHA*similarity_with_annotated_frame)/self.gram_matrix[0, 0] > self.similarity_bound

            # return np.array(similarities).median() > self.similarity_bound

        elif 'annotated and max' == self.method_LT_memory:
            ALPHA = self.alpha_A_L_A_M
            print(ALPHA)
            print(self.similarity_bound)
            print(((1-ALPHA)*np.array(similarities).max() + ALPHA*similarity_with_annotated_frame)/self.gram_matrix[0, 0])
            print(np.array(similarities).max()/self.gram_matrix[0, 0])
            print(similarity_with_annotated_frame/self.gram_matrix[0, 0])
            print(((1-ALPHA)*np.array(similarities).max() + ALPHA*similarity_with_annotated_frame)/self.gram_matrix[0, 0] > self.similarity_bound)
            return ((1-ALPHA)*np.array(similarities).max() + ALPHA*similarity_with_annotated_frame)/self.gram_matrix[0, 0] > self.similarity_bound

            # return np.array(similarities).median() > self.similarity_bound

        # # Not used at the moment
        # elif 'ensemble lower bound' == self.method_LT_memory:
        #     # print(self._lt_gram_matrix)
        #     ensemble_similarities = [compute_similarity_between_two_features(key[obj_idx], self.memory_keys[
        #         obj_idx]).cpu().detach().numpy() for obj_idx in range(0, nbr_of_objects)]
        #     # elif "values" == use_keys_or_values:
        #     #     ensemble_similarities = [compute_similarity_between_two_features(prev_value[obj_idx],self.LT_memory_values[obj_idx]).cpu().detach().numpy()  for obj_idx in range(0, self.k)]
        #     if self.gram_matrix.shape != ():
        #         return (
        #                     np.array(ensemble_similarities).mean(axis=0).squeeze() > self.similarity_bound * np.diag(
        #                 self.gram_matrix)).min()
        #     else:
        #         return np.array(ensemble_similarities).mean(
        #             axis=0).squeeze() > self.similarity_bound * self.gram_matrix.shape



    def _condition_for_only_considering_similar_frames_values_too(self, key, value, strided_center, strided_box, ST_gamma_diversity = 0):
        # Check with the similarity of the current frame and the ones in the LT memory.
        # If similarity too low, might be because too much background taken into account.
        values,indexes,counts = np.unique(np.array(self.frame_indexes_list),True,False,True)

        # print('values,indexes,counts')
        # print(values,indexes,counts)

        # Just the clone, copy the affinity of the annotated frame and give it to the other also! ATTENTION ! ONLY FOR THE INITIALIZATION !!
        for v, i, c in zip(values, indexes,counts):  # This only works at the initialization and is not a general solution. !!!!!
            if 1 == c: continue

            affinity_to_clone = self.affinity_matrices[i,:,:].clone() # Take the index 0 because its the annotated variant
            new_affinities = torch.zeros(c-1,*affinity_to_clone.shape).cuda()
            for j in range(0,c-1):
                new_affinities[j]+=affinity_to_clone

            self.affinity_matrices = torch.cat([new_affinities, self.affinity_matrices], dim=0) # Add at the beginning of the extract affinity matrices, as you want to add affiniies only for the annotation frame

        # Create temporary_affinity_matrix that matches the number of elements in the memory. Very important to be constistent
        # with the number of frames stored in the lt memory and the affinity matrices.

        similarities = self._compute_similarities_with_memory_values_too(strided_center, strided_box, key, value, self.memory_keys, self.memory_values, self.affinity_matrices, self.Use_affinity, self.Use_tukey_window, self.tukey_alpha)
        similarity_with_annotated_frame = similarities[0]

        # See if second condition is valid: That the similarity is good enough
        # print(similarity_with_annotated_frame)
        # print(self.similarity_bound)
        if 'static lower bound' == self.method_LT_memory:
            if self.normalizing_factor_gram_matrix is None:
                return similarity_with_annotated_frame > self.similarity_bound
            else:
                return similarity_with_annotated_frame/self.normalizing_factor_gram_matrix > self.similarity_bound

        elif 'dynamic lower bound' == self.method_LT_memory:
            return similarity_with_annotated_frame > self.similarity_bound - ST_gamma_diversity

        elif 'ensemble lower bound' == self.method_LT_memory:
            # print(similarities)
            if self.normalizing_factor_gram_matrix is None:
                return np.array(similarities).mean() > self.similarity_bound
            else:
                return np.array(similarities).mean()/self.normalizing_factor_gram_matrix > self.similarity_bound

        elif 'ensemble lower boundtwo' == self.method_LT_memory:
            # print(similarities)
            if self.normalizing_factor_gram_matrix is None:
                return np.array(similarities).max() > self.similarity_bound
            else:
                return np.array(similarities).max() / self.normalizing_factor_gram_matrix > self.similarity_bound
            # return np.array(similarities).median() > self.similarity_bound
        # # Not used at the moment
        # elif 'ensemble lower bound' == self.method_LT_memory:
        #     # print(self._lt_gram_matrix)
        #     ensemble_similarities = [compute_similarity_between_two_features(key[obj_idx], self.memory_keys[
        #         obj_idx]).cpu().detach().numpy() for obj_idx in range(0, nbr_of_objects)]
        #     # elif "values" == use_keys_or_values:
        #     #     ensemble_similarities = [compute_similarity_between_two_features(prev_value[obj_idx],self.LT_memory_values[obj_idx]).cpu().detach().numpy()  for obj_idx in range(0, self.k)]
        #     if self.gram_matrix.shape != ():
        #         return (
        #                     np.array(ensemble_similarities).mean(axis=0).squeeze() > self.similarity_bound * np.diag(
        #                 self.gram_matrix)).min()
        #     else:
        #         return np.array(ensemble_similarities).mean(
        #             axis=0).squeeze() > self.similarity_bound * self.gram_matrix.shape


    def _condition_to_update_LT_Mem_only_if_diversity_is_enhanced(self, key):
        # Determine if the volume of the parallelotope gets bigger by including the next candidate frame in the long-term memory.
        self._compute_gram_det()

        # print('*********************')
        # print(self.gram_matrix)
        # print(self._compute_det(self.gram_matrix))
        # print('*********************')

        starting_position = 1 if self.keep_annotated_frame_in_LT_memory else 0
        temporary_determinants_list = []
        temporary_gram_matrix_list = []

        # Check the first value of the gram matrix, if not equal to one then take that value and use it for the remaining of the sequence
        # if self.normalizing_factor_gram_matrix is None and 1!= self.gram_matrix[0,0]:
        #     self.normalizing_factor_gram_matrix = self.gram_matrix[0,0]

        denominateur = self.gram_matrix[0,0]

        for idx in range(starting_position, self.max_size_of_memory):
            temporary_mem = torch.cat((self.memory_keys[:,:,:idx], self.memory_keys[:,:,idx+1:]), dim=2)
            temporary_affinity = torch.cat((self.affinity_matrices[:idx,:,:],self.affinity_matrices[idx+1:,:,:]),dim=0)
            temporary_gram_matrix = self.gram_matrix.copy()

            temporary_similarities = self._compute_similarities(self.use_cosine_sim, key, temporary_mem, temporary_affinity, self.Use_affinity, self.Use_tukey_window, self.tukey_alpha)
            temporary_gram_matrix = self._delete_col_N_row_from_Gram_matrix(temporary_gram_matrix, idx)
            # print('temporary_gram_matrix')
            # print(temporary_gram_matrix)
            # print('temporary_similarities')
            # print(temporary_similarities)
            temporary_gram_matrix = self._update_gram_matrix(temporary_gram_matrix, temporary_similarities)
            temporary_determinant = self._compute_det(temporary_gram_matrix,denominateur)

            # print('temporary_similarities',temporary_similarities)

            temporary_gram_matrix_list.append(temporary_gram_matrix)
            temporary_determinants_list.append(temporary_determinant)

            # print(temporary_gram_matrix)
            # print(temporary_determinant)

        # print('*********************')

        # Check if determinant is bigger then the base determinant
        best_idx = np.argmax(temporary_determinants_list)
        best_temporary_det = temporary_determinants_list[best_idx]
        # print('self.gram_det', self.gram_det)
        # print('best_temporary_det', best_temporary_det)
        # Cond = self.gram_det <= best_temporary_det
        # print('self.gram_det <= best_temporary_det', Cond)

        print(best_temporary_det)
        print(self.gram_det)

        if self.gram_det <= best_temporary_det:
            self.gram_matrix = temporary_gram_matrix_list[best_idx]
            best_idx += starting_position
            # print('starting_position', starting_position)
            # print('best_idx', best_idx)

            # ic(self.gram_det)
            return True, best_idx
        else:
            return False, None
        # return True, 0  # simulating Short-term memory


    def _condition_to_update_LT_Mem_only_if_diversity_is_enhanced_values_too(self, key, value, strided_center, strided_box):
        # Determine if the volume of the parallelotope gets bigger by including the next candidate frame in the long-term memory.
        self._compute_gram_det()

        # print('*********************')
        # print(self.gram_matrix)
        # print(self._compute_det(self.gram_matrix))
        # print('*********************')

        starting_position = 1 if self.keep_annotated_frame_in_LT_memory else 0
        temporary_determinants_list = []
        temporary_gram_matrix_list = []

        # Check the first value of the gram matrix, if not equal to one then take that value and use it for the remaining of the sequence
        # if self.normalizing_factor_gram_matrix is None and 1!= self.gram_matrix[0,0]:
        #     self.normalizing_factor_gram_matrix = self.gram_matrix[0,0]

        for idx in range(starting_position, self.max_size_of_memory):
            temporary_mem = torch.cat((self.memory_keys[:,:,:idx], self.memory_keys[:,:,idx+1:]), dim=2)
            temporary_val = torch.cat((self.memory_values[:, :, :idx], self.memory_values[:, :, idx + 1:]), dim=2)
            temporary_affinity = torch.cat((self.affinity_matrices[:idx,:,:],self.affinity_matrices[idx+1:,:,:]),dim=0)
            temporary_gram_matrix = self.gram_matrix.copy()

            temporary_similarities = self._compute_similarities_with_memory_values_too(strided_center, strided_box, key, value, temporary_mem, temporary_val, temporary_affinity, self.Use_affinity, self.Use_tukey_window, self.tukey_alpha)
            temporary_gram_matrix = self._delete_col_N_row_from_Gram_matrix(temporary_gram_matrix, idx)
            # print('temporary_gram_matrix')
            # print(temporary_gram_matrix)
            # print('temporary_similarities')
            # print(temporary_similarities)
            temporary_gram_matrix = self._update_gram_matrix(temporary_gram_matrix, temporary_similarities)
            if self.normalizing_factor_gram_matrix is not None:
                temporary_gram_matrix = temporary_gram_matrix/self.normalizing_factor_gram_matrix
            temporary_determinant = self._compute_det(temporary_gram_matrix)

            # print('temporary_similarities',temporary_similarities)

            temporary_gram_matrix_list.append(temporary_gram_matrix)
            temporary_determinants_list.append(temporary_determinant)

        #     print(temporary_gram_matrix)
        #     print(temporary_determinant)
        #
        # print('*********************')

        # Check if determinant is bigger then the base determinant
        best_idx = np.argmax(temporary_determinants_list)
        best_temporary_det = temporary_determinants_list[best_idx]
        # print('self.gram_det', self.gram_det)
        # print('best_temporary_det', best_temporary_det)
        # Cond = self.gram_det <= best_temporary_det
        # print('self.gram_det <= best_temporary_det', Cond)
        if self.gram_det <= best_temporary_det:
            self.gram_matrix = temporary_gram_matrix_list[best_idx]
            best_idx += starting_position
            # print('starting_position', starting_position)
            # print('best_idx', best_idx)

            # ic(self.gram_det)
            return True, best_idx
        else:
            return False, None
        # return True, 0  # simulating Short-term memory


    def _condition_to_update_LT_Mem_only_if_diversity_is_enhanced_crop_mode_3(self, feature_f16):
        # Determine if the volume of the parallelotope gets bigger by including the next candidate frame in the long-term memory.
        self._compute_gram_det()

        # print('*********************')
        # print(self.gram_matrix)
        # print(self._compute_det(self.gram_matrix))
        # print('*********************')

        starting_position = 1 if self.keep_annotated_frame_in_LT_memory else 0
        temporary_determinants_list = []
        temporary_gram_matrix_list = []

        # print('----------------------------')
        # print(self.memory_f16.shape)
        # print('----------------------------')

        for idx in range(starting_position, self.max_size_of_memory):
            temporary_mem = torch.cat((self.memory_f16[:idx], self.memory_f16[idx+1:]), dim=0)
            temporary_gram_matrix = self.gram_matrix.copy()

            temporary_similarities = self._compute_similarities_mode_3(feature_f16, temporary_mem, self.Use_tukey_window, self.tukey_alpha)
            temporary_gram_matrix = self._delete_col_N_row_from_Gram_matrix(temporary_gram_matrix, idx)
            # print('temporary_gram_matrix')
            # print(temporary_gram_matrix)
            # print('temporary_similarities')
            # print(temporary_similarities)
            # print(feature_f16.shape)
            # print(temporary_mem.shape)
            temporary_gram_matrix = self._update_gram_matrix(temporary_gram_matrix, temporary_similarities)
            temporary_determinant = self._compute_det(temporary_gram_matrix)

            # print('temporary_similarities',temporary_similarities)

            temporary_gram_matrix_list.append(temporary_gram_matrix)
            temporary_determinants_list.append(temporary_determinant)

        #     print(temporary_gram_matrix)
        #     print(temporary_determinant)
        #
        # print('*********************')

        # Check if determinant is bigger then the base determinant
        best_idx = np.argmax(temporary_determinants_list)
        best_temporary_det = temporary_determinants_list[best_idx]
        # print('self.gram_det', self.gram_det)
        # print('best_temporary_det', best_temporary_det)
        # Cond = self.gram_det <= best_temporary_det
        # print('self.gram_det <= best_temporary_det', Cond)
        if self.gram_det <= best_temporary_det:
            self.gram_matrix = temporary_gram_matrix_list[best_idx]
            best_idx += starting_position
            # print('starting_position', starting_position)
            # print('best_idx', best_idx)

            # ic(self.gram_det)
            return True, best_idx
        else:
            return False, None
        # return True, 0  # simulating Short-term memory


    @property
    def is_LT_Mem_updated(self):
        return self.updated_LT_Mem_flag

