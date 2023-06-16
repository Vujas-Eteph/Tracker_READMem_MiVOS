"""
This file can handle DAVIS 2016/2017 evaluation.
"""

import torch
import numpy as np
import cv2
import torch.nn.functional as F
from MiVOS.model.propagation.prop_net import PropagationNetwork
from model.aggregate import aggregate_wbg
from util.tensor_util import pad_divide_by

# Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ
# modified by Stéphane Vujasinovic is enclaved in such manner
from icecream import ic
import yaml
import os
import sys

# ic(os.getcwd())
from HoliMem.HoliMem import HoliMem

from atomic_crop import super_crop, uncrop_mask, extract_size_of_target_for_window_filtering
import torchvision.transforms as T
from PIL import Image

transform = T.ToPILImage()

import lovely_tensors as lt


from Toolbox_Eteph.Debugging.ST_LT_memory_plot.Plot_ST_and_LT_Memory import *
# Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ Σ


class InferenceCore:
    def __init__(self, prop_net: PropagationNetwork, num_objects, mem_config:str, debugging_flag=False, record_det_flag=False, use_super_crop=False):
        self.prop_net = prop_net
        self.device = 'cuda'
        self.k = num_objects

        # Initialize HoliMem attributes
        self.HoliMem = HoliMem(mem_config, debugging_flag=True)
        self.HoliMem.nbr_of_objects_working_with = num_objects

        # Initialize flags for helping by the debugging process
        self.debugging_flag = debugging_flag
        self.record_det_flag = record_det_flag
        # self.use_super_crop = True if use_super_crop !=1.0 else False ## TODO SET A VARIABLE THAT ALLOWS ME TO CHOSE THIS
        # self.use_super_crop = False ## TODO SET A VARIABLE THAT ALLOWS ME TO CHOSE THIS
        # if self.use_super_crop:



        self.Affinity_mode = 0  # Affinity 0 is the one with query keys and memory keys, for STCN this does not matter
        self.mode = 1           # mode 1 default , 2 padding, 3 crops
        # self.mode = 3           # mode 1 default , 2 padding, 3 crop + padding based on THOR
        if 3 == self.mode:
            self.hypo_coeff = 1.44
        # self.use_cosine_sim = True # True default, if False use dot product normalized on the annotated frame
        # self.use_cosine_sim = False # True default, if False use dot product normalized on the annotated frame

        # self.use_cosine_sim = True

    def get_path_2_image_folder(self, img_folder: str):
        self.path_2_image_folder = img_folder


    def unpad(self, data, pad):
        if len(data.shape) == 4:
            if pad[2] + pad[3] > 0:
                data = data[:, :, pad[2]:-pad[3], :]
            if pad[0] + pad[1] > 0:
                data = data[:, :, :, pad[0]:-pad[1]]
        elif len(data.shape) == 3:
            if pad[2] + pad[3] > 0:
                data = data[:, pad[2]:-pad[3], :]
            if pad[0] + pad[1] > 0:
                data = data[:, :, pad[0]:-pad[1]]
        else:
            raise NotImplementedError
        return data


    def _get_query_kv_buffered(self, image):
        # not actually buffered
        return self.prop_net.get_query_values(image.cuda())
        # f16, f8, f4, k16, v16 = self.prop_net.get_query_values(image.cuda())
        # _, _, h, w = f16.size()
        # pre_mask, _ = torch.max(self.prob[1:], dim=0)
        # pre_mask = pre_mask.unsqueeze(0)
        # pre_mask = F.interpolate(pre_mask, size=[h, w], mode='bilinear')
        # concat_f16 = torch.cat([f16, pre_mask], dim=1)
        # concat_f16 = self.prop_net.concat_conv(concat_f16)
        # concat_f16 = torch.sigmoid(concat_f16)
        # concat_f16 = f16 * concat_f16
        # k16, v16 = self.prop_net.kv_q_f16(concat_f16)
        # result = (concat_f16, f8, f4, k16, v16)
        # return result


    def _set_image(self, sequence_length, OG_image):
        # True dimensions
        OG_image = OG_image.unsqueeze(dim=0).cuda()
        # self.t = sequence_length
        self.image, self.pad = pad_divide_by(OG_image, 16)


    def set_annotated_frame(self, idx, sequence_length, image, anno_mask):
        self.HoliMem.reset_HoliMem()    # Reset the ST and LT memories

        # print(image.shape)
        self._set_image(sequence_length, image)

        self.annotated_image = self.image.clone()

        anno_mask = anno_mask.unsqueeze(dim=1)
        mask, _ = pad_divide_by(anno_mask.cuda(), 16)
        self.prob = aggregate_wbg(mask, keep_bg=True)

        # print(anno_mask.shape)
        # print(mask.shape)
        # print(self.prob.shape)


        # # CROP the IMAGE region ?
        # if not self.use_super_crop:
        #     self.prob = aggregate_wbg(mask, keep_bg=True)
        # else:
        #     # CROP AND RESIZE
        #     self.IMG_RESOLUTION = self.image.shape[-2:]
        #     # print(self.IMG_RESOLUTION)
        #     self.image, crop_prob, _ , _= super_crop(self.image.clone(), self.prob.clone(), self.k, self.hypo_coeff, None)
        #     # self.prev_mask_to_use_for_crops = self.mask
        #
        #     # img_PIL = transform(self.image.clone().squeeze())
        #     # img_PIL.show()
        #
        #     self.prob = crop_prob


        # KV pair for the interacting frame
        print(self.image.shape)
        print(self.prob.shape)
        # anno_key_k, anno_key_v = self.prop_net.memorize(self.image.cuda(),self.prob[1:].cuda())
        # Add the annotated frame to the ST and LT Memory
        if 1 == self.mode:
            anno_key_k, anno_key_v = self.prop_net.memorize(self.image.cuda(), self.prob[1:].cuda())
            self.HoliMem.update_HoliMem(idx, anno_key_k, anno_key_v)
        # elif 3 == self.mode:
        #     # self.idx_for_mode_3 = -2  # 0 for features from the bakcbone ResNet or -2 for keys
        #     self.idx_for_mode_3 = 0  # 0 for features from the bakcbone ResNet or -2 for keys
        #     self.IMG_RESOLUTION = self.image.shape
        #     crop_img, crop_prob, crop_vector, pad_vector = super_crop(self.image.clone(), self.prob, self.k, self.hypo_coeff, None)
        #
        #     c_query = self._get_query_kv_buffered(crop_img)
        #     feature_f16 = c_query[self.idx_for_mode_3]
        #     anno_key_k, anno_key_v = self.prop_net.memorize(crop_img.cuda(), crop_prob[1:].cuda())
        #     # self.HoliMem.update_HoliMem_based_on_crop_mode_3(idx, anno_key_k, anno_key_v, feature_f16)
        #     self.HoliMem.update_HoliMem(idx, anno_key_k, anno_key_v)
        #
        #     self.prob = uncrop_mask(crop_prob, crop_vector, pad_vector, self.IMG_RESOLUTION)


        # if not self.use_cosine_sim:
        #     self.HoliMem.LT_HoliMem.reset_LT_gram_matrix(anno_key_k)


        return self.unpad(self.prob,self.pad)


    def _adapt_img(self,OG_image):
        return pad_divide_by(OG_image.unsqueeze(dim=0).cuda(), 16)


    def step(self, idx, image):
        print('idx:',idx)
        # print(image.shape)
        # Extract the key and values of the current frame
        img, pad = self._adapt_img(image)

        ic('check1')

        # if idx == 3237:
        #     transform(img[0]).show()

        # print(img)
        ori_img = img.clone()

        # CROP the IMAGE region ?
        # if 3 == self.mode:
        #     img, crop_prob, crop_vector, pad_vector = super_crop(ori_img.clone(), self.prob.clone(), self.k, self.hypo_coeff, None)

        # if idx == 3237:
        #     print(img.shape)
        #     transform(img[0]).show()
        #     print('hi')


        query = self._get_query_kv_buffered(img)

        ic('check2')

        # Extract the holistic representation
        HoliMem_idx_list, HoliMem_keys, HoliMem_values = self.HoliMem.get_holistic_memory()

        ic('check3')

        # Infer the segmentation mask based on the holistic representation stored in the memory
        Holi_out_mask = self.prop_net.segment_with_query(HoliMem_keys, HoliMem_values, *query)

        Holi_out_mask = aggregate_wbg(Holi_out_mask, keep_bg=True)
        self.prob = Holi_out_mask

        ic('check4')

        # self.display_an_embending(self.prob)# Display memory values or others

        # if self.use_super_crop:
        #     self.prev_mask_to_use_for_crops = uncrop(Holi_out_mask.clone(), crop_vector, self.IMG_RESOLUTION)
        #     # self.prob = uncrop(self.prob.clone(), crop_vector, self.IMG_RESOLUTION)



        # Extract the features of the current frame through the memory network
        # prev_key, prev_value = self.prop_net.memorize(img, Holi_out_mask[1:])
        # print(Holi_out_mask.shape)
        # prev_key, prev_value = self.prop_net.memorize(ori_img, Holi_out_mask[1:])

        # # Update the holistic memory
        # if 0 == self.Affinity_mode:
        #     self.HoliMem.set_affinity_matrices(self.prop_net.get_affinity())
        # elif 1 == self.Affinity_mode:
        #     Ddd = self.prop_net.compute_similarites_for_memory_key_types(HoliMem_keys,prev_key)
        #     self.HoliMem.set_affinity_matrices(Ddd)


        # TESTING !!
        if 1 == self.mode:
            prev_key, prev_value = self.prop_net.memorize(ori_img, Holi_out_mask[1:])
            # print(prev_key.shape)

            ic('check5')
            # Update the holistic memory
            if 0 == self.Affinity_mode:
                self.HoliMem.set_affinity_matrices(self.prop_net.get_affinity())
            elif 1 == self.Affinity_mode:
                Ddd = self.prop_net.compute_similarites_for_memory_key_types(HoliMem_keys, prev_key)
                self.HoliMem.set_affinity_matrices(Ddd)

            ic('check6')

            self.HoliMem.update_HoliMem(idx, prev_key, prev_value)
        # elif 2 == self.mode:
        #     ###########################################
        #     # crop_region = extract_size_of_target_for_window_filtering(self.prob,1.0)
        #     c_img, _, crop_vector = super_crop(img.clone(), self.prob, self.k,
        #                                        1.1, None)
        #     c_prev_key, c_prev_value = self.prop_net.memorize(c_img, Holi_out_mask[1:])
        #     ###########################################
        #     self.HoliMem.find_best_LT_idx(idx, prev_key, prev_value, c_prev_key, c_prev_value)
        # elif 3 == self.mode:
        #     print(img.shape)
        #     self.prob = uncrop_mask(self.prob, crop_vector, pad_vector, self.IMG_RESOLUTION)
        #
        #     img, crop_prob, crop_vector, pad_vector = super_crop(ori_img.clone(), self.prob, self.k, self.hypo_coeff, None)
        #
        #
        #     prev_key, prev_value = self.prop_net.memorize(img.cuda(), crop_prob[1:].cuda())
        #     c_query = self._get_query_kv_buffered(img)
        #
        #     # print(c_query[0].shape) # Directly the feature space
        #     # print(c_query[-2].shape) # Using the key vectors
        #
        #     feature_f16 = c_query[self.idx_for_mode_3]
        #     # self.HoliMem.update_HoliMem_based_on_crop_mode_3(idx, prev_key, prev_value, feature_f16)
        #     self.HoliMem.update_HoliMem(idx, prev_key, prev_value)#, feature_f16)

        ic('check7')


        return self.unpad(self.prob, pad)




    @property
    def return_lt_det(self):
        return self.HoliMem.LT_gram_det.copy()

    @property
    def ST_N_LT_Memories(self):
        return self.HoliMem.ST_Memory_indexes.copy(), self.HoliMem.LT_Memory_indexes.copy()

    @ST_N_LT_Memories.setter
    def ST_N_LT_Memories(self, new_st_indexes, new_lt_indexes):
        self.HoliMem.ST_Memory_indexes = new_st_indexes
        self.HoliMem.LT_Memory_indexes = new_lt_indexes

    @property
    def get_size_of_ST_N_LT_memory(self):
        return self.HoliMem.get_size_of_ST_N_LT_memory


    def display_an_embending(self, input):
        # Take a look at the memory value of the annotated frame
        print(input)
        print(input.shape)

        # numpy_memory_value = input[0,:,0].clone().permute(1,2,0).detach().cpu().numpy()
        numpy_memory_value = input[:,0].clone().permute(1,2,0).detach().cpu().numpy()[:,:,1]# take the obj, or 0 for the background
        print(numpy_memory_value.shape)

        # np_2_img = numpy_memory_value.std(axis=2)
        np_2_img = numpy_memory_value
        # np_2_img = np_2_img*255
        # np_2_img = np_2_img - np_2_img.min()
        # np_2_img = np_2_img/(np_2_img.max())*255
        # np_2_img = numpy_memory_value.min(axis=2)
        # np_2_img = numpy_memory_value.std(axis=2)
        print(np_2_img.shape)
        resized = cv2.resize(np_2_img, (912,480), interpolation=cv2.INTER_AREA)

        #
        while True:
            cv2.imshow('Memory_value', resized)#.astype(np.int8))

            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)
            if cv2.key == ord('q'):
                # closing all open windows
                cv2.destroyAllWindows()
                break

        print('hi')
