import numpy as np
import torch
import torchvision

import lovely_tensors as lt
from icecream import ic

# lt.monkey_patch()

def extract_size_of_target_for_window_filtering(msk:torch.Tensor, hypo):
    coords_d = {}
    coords_d_ext = {}
    for i, m in enumerate(msk[1:].round()):
        # print(i)
        # print(m.round())
        # print(m)
        coords = extract_coord_for_crop(m[0])
        coords_d.update({f'{i+1}':coords})
        ext_coords = find_search_region(coords,hypo) # [new_y_min, new_y_max, new_x_min, new_x_max]
        # ic(coords)
        # ic(ext_coords)
        ext_coords[:2] = torch.clip(ext_coords[:2],0,msk.shape[3])
        ext_coords[2:] = torch.clip(ext_coords[2:],0,msk.shape[2])
        # ic(ext_coords)
        coords_d_ext.update({f'{i+1}':ext_coords})
        # ic(m.shape)

    y_min = None
    y_max = None
    x_min = None
    x_max = None

    for k,v in coords_d_ext.items():
        # print(v)
        y_min = v[0] if y_min is None else min(y_min,v[0])
        y_max = v[1] if y_max is None else max(y_max,v[1])
        x_min = v[2] if x_min is None else min(x_min,v[2])
        x_max = v[3] if x_max is None else max(x_max,v[3])

    crop_vector = [y_min.item(),y_max.item(),x_min.item(),x_max.item()] # y_min, y_max, x_min, x_max

    return crop_vector


def uncrop_mask(tsr:torch.Tensor, crop_vector:list, pad_vector:list, ORIGINAL_RESOLUTION: list):
    # print(tsr) # mask [1,1,H,W]
    # print(crop_vector)
    # print(pad_vector)
    # print(ORIGINAL_RESOLUTION)

    size_before_resizing = np.array(crop_vector)
    size_before_resizing = [int(np.diff(size_before_resizing[2:])[0]),int(np.diff(size_before_resizing[:2])[0])]
    print(size_before_resizing)

    # Resize the image
    resize_tsr = torch.nn.functional.interpolate(tsr,
                                                 size=size_before_resizing,
                                                 mode='bilinear',
                                                 align_corners=True,
                                                 antialias=True)

    # print(resize_tsr.shape)

    # Cut the padding part
    unpadded_tsr = resize_tsr[:,:,
                   int(pad_vector[-2]):int(size_before_resizing[0] - pad_vector[-1]),
                   int(pad_vector[0]):int(size_before_resizing[1] - pad_vector[1])]

    # print(unpadded_tsr.shape)

    # Place the output correctly back to the orignal size

    # print(tsr.min(), tsr.max())
    complete_IMG = torch.zeros([tsr.shape[0], 1, ORIGINAL_RESOLUTION[-2], ORIGINAL_RESOLUTION[-1]])
    for idx in range(0,tsr.shape[0]):
        if idx == 0:
            complete_img = torch.ones([1, 1, ORIGINAL_RESOLUTION[-2], ORIGINAL_RESOLUTION[-1]])
        else:
            complete_img = torch.zeros([1, 1, ORIGINAL_RESOLUTION[-2], ORIGINAL_RESOLUTION[-1]])

        # print(idx)
        # print(unpadded_tsr[idx].shape)
        # print(crop_vector)
        complete_img[:,:,
        int(max(0,crop_vector[2])):int(min(ORIGINAL_RESOLUTION[-1],crop_vector[3])),
        int(max(0,crop_vector[0])):int(min(ORIGINAL_RESOLUTION[-1],crop_vector[1]))] = unpadded_tsr[idx].unsqueeze(dim=0)

        complete_IMG[idx] = complete_img

    # print(complete_IMG.shape)

    return complete_IMG








    # if crop_vector is None: return tsr
    #
    # print(ORIGINAL_RESOLUTION)
    # print(crop_vector)
    # print(tsr.shape)
    #
    # diff = lambda vector : int(torch.floor(abs(vector[0] - vector[1])))
    # vec = lambda vector : np.array([diff(vector[0:2]),diff(vector[2:4])]) # [new_y_min, new_y_max, new_x_min, new_x_max] -> [y_height, x_width]
    # croped_size = vec(crop_vector)
    # print(croped_size)
    #
    # resized_crop_tsr = torch.nn.functional.interpolate(tsr,
    #                                                    size=[croped_size[1],croped_size[0]],
    #                                                    mode='bicubic',
    #                                                    align_corners=True,
    #                                                    antialias=True)
    #
    # # Placing crop_resized part back in the original resolution
    # ZERO_tensor = torch.zeros([*tsr.shape[:2],*ORIGINAL_RESOLUTION])
    # y_min, y_max, x_min, x_max = crop_vector
    # ZERO_tensor[:, :, int(x_min.item()):int(x_max.item()), int(y_min.item()):int(y_max.item())] = resized_crop_tsr
    #
    # print(ZERO_tensor.shape)
    #
    # return ZERO_tensor



def super_crop(img:torch.Tensor, msk:torch.Tensor, nbr_oo: int, hypo=1, annotated_image=None, real_crop = False):
    '''Crops the image, h how much to multiply the Hypotenuse, nbr_oo: number of objects'''
    # discard_the_background, and for every object find the coordinates of the crop
    # ic(msk.shape)
    # msk = msk[1:]
    # RESIZE_IMG_SIZE = 512
    # RESIZE_IMG_SIZE = 480
    RESIZE_IMG_SIZE = 256
    coords_d = {}
    coords_d_ext = {}
    # ic(img.shape)
    for i, m in enumerate(msk[1:].round()):
        # print(i)
        # print(m.round())
        # print(m)
        coords = extract_coord_for_crop(m[0])
        coords_d.update({f'{i+1}':coords})
        ext_coords = find_search_region(coords,hypo) # [new_y_min, new_y_max, new_x_min, new_x_max]
        # ic(coords)
        # ic(ext_coords)
        # print(ext_coords)
        # ext_coords[:2] = torch.clip(ext_coords[:2],0,img.shape[3])
        # ext_coords[2:] = torch.clip(ext_coords[2:],0,img.shape[2])
        # ic(ext_coords)
        coords_d_ext.update({f'{i+1}':ext_coords})
        # ic(m.shape)

    print(f'coord_d_ext{coords_d_ext}')

    # ic(nbr_oo)
    # ic(coords_d_ext)
    # ic(img.shape)
    # ic(msk.shape)
    # pass

    # print(coords_d_ext)

    y_min = None
    y_max = None
    x_min = None
    x_max = None

    for k,v in coords_d_ext.items():
        print(v)
        # y_min = v[0] if y_min is None else min(y_min,v[0])
        # y_max = v[1] if y_max is None else max(y_max,v[1])
        # x_min = v[2] if x_min is None else min(x_min,v[2])
        # x_max = v[3] if x_max is None else max(x_max,v[3])
        y_min = v[2] if y_min is None else min(y_min,v[2])
        y_max = v[3] if y_max is None else max(y_max,v[3])
        x_min = v[0] if x_min is None else min(x_min,v[0])
        x_max = v[1] if x_max is None else max(x_max,v[1])

    # print(ext_coords)

    # y_min = ext_coords[2]
    # y_max = ext_coords[3]
    # x_min = ext_coords[0]
    # x_max = ext_coords[1]

    # y_min = ext_coords[0]
    # y_max = ext_coords[1]
    # x_min = ext_coords[2]
    # x_max = ext_coords[3]

    # check the values at least over 10 by 10
    _h = abs(y_min - y_max)
    _w = abs(x_min - x_max)

    if _h < 10:
        y_min -= 5
        y_max += 5
    if _w < 10:
        x_min -= 5
        x_max += 5




    crop_vector = [x_min,x_max,y_min,y_max]

    print('crop_vector',crop_vector)
    # crop the image
    crop_img = img.clone()
    # print(crop_img.shape)
    # print(y_min)
    # print(y_max)
    # print(x_min)
    # print(x_max)

    # padding:
    x_pad_left = abs(x_min) if x_min < 0 else 0
    x_pad_right = abs(x_max-img.shape[-1]) if x_max>img.shape[-1] else 0
    y_pad_top = abs(y_min) if y_min < 0 else 0
    y_pad_bottom = abs(y_max-img.shape[-2]) if y_max>img.shape[-2] else 0

    # print(x_pad_left)
    # print(x_pad_right)
    # print(y_pad_top)
    # print(y_pad_bottom)
    pad_vector = [x_pad_left,x_pad_right,y_pad_top,y_pad_bottom]
    print('pad_vector',pad_vector)

    # print(int(max(0, y_min)))
    # print(int(min(img.shape[-2],y_max)))

    crop_img = crop_img[:,:,int(max(0,y_min)):int(min(img.shape[-2],y_max)),
               int(max(0,x_min)):int(min(img.shape[-1],x_max))]
    # print(crop_img.shape)

    # crop the mask
    crop_msk = msk.clone()
    crop_msk = crop_msk[:,:,int(max(0,y_min)):int(min(img.shape[-2],y_max)),
               int(max(0,x_min)):int(min(img.shape[-1],x_max))]

    # Add padding:
    crop_pad_img = torch.zeros([crop_img.shape[0],
                                crop_img.shape[1],
                                int(crop_img.shape[2]+y_pad_top+y_pad_bottom),
                                int(crop_img.shape[3]+x_pad_left+x_pad_right)])
    crop_pad_msk = torch.zeros([crop_msk.shape[0],
                                crop_msk.shape[1],
                                int(crop_msk.shape[2]+y_pad_top+y_pad_bottom),
                                int(crop_msk.shape[3]+x_pad_left+x_pad_right)])

    # print(crop_pad_img.shape)

    # Integrate the image on the crop_padded region
    crop_pad_img[:,:,
    int(y_pad_top):int(crop_pad_img.shape[-2]-y_pad_bottom),
    int(x_pad_left):int(crop_pad_img.shape[-1]-x_pad_right)] = crop_img

    print(crop_img.shape)
    print(crop_msk.shape)
    crop_pad_msk[:,:,
    int(y_pad_top):int(crop_pad_img.shape[-2]-y_pad_bottom),
    int(x_pad_left):int(crop_pad_img.shape[-1]-x_pad_right)] = crop_msk

    # Resize the image
    crop_resize_img = torch.nn.functional.interpolate(crop_pad_img,
                                                      size=RESIZE_IMG_SIZE,
                                                      mode='bilinear',
                                                      align_corners=True,
                                                      antialias=True)

    crop_resize_msk = torch.nn.functional.interpolate(crop_pad_msk,
                                               size=RESIZE_IMG_SIZE,
                                               mode='bilinear',
                                               align_corners=True,
                                               antialias=True)

    # print(crop_img)
    # print(crop_msk)
    # print(crop_pad_img)
    # print(crop_pad_msk)
    # print(crop_resize_img)
    # print(crop_resize_msk)
    return crop_resize_img, crop_resize_msk, crop_vector, pad_vector

    # # use padding or crop ??
    #
    # print(img.shape)
    # print(y_min,y_max,x_min,x_max)
    # print(crop_img.shape)
    #
    #
    # if real_crop:
    #     # Resize to a fixes size:
    #     # 480 x 480
    #     print(crop_img)
    #     print(crop_img.shape)
    #     print(crop_img.nelement())
    #     print(torch.empty([]).nelement())
    #
    #     # Check that we add padding before rescaling on the sides that need it.
    #
    #
    #
    #
    #     if 0 in crop_img.shape: # in case no prediction was made use the complete image
    #         crop_img = img.clone()
    #         crop_msk = msk.clone()
    #         print(crop_img.shape)
    #         print('ACTUALLY GOT IN...')
    #
    #
    #     crop_resize_img = torch.nn.functional.interpolate(crop_img,
    #                                                       size=RESIZE_IMG_SIZE,
    #                                                       mode='bicubic',
    #                                                       align_corners = True,
    #                                                       antialias = True)
    #
    #
    #     crop_msk = torch.nn.functional.interpolate(crop_msk,
    #                                                size=RESIZE_IMG_SIZE,
    #                                                mode='bicubic',
    #                                                align_corners = True,
    #                                                antialias = True)
    #
    #     print(crop_resize_img.shape)
    #
    #     return crop_resize_img, crop_msk, crop_vector, None
    # else:
    #     if annotated_image is None:
    #         ZERO_img = torch.zeros([*img.shape])
    #     else:
    #     #     print(annotated_image.shape)
    #     #     print(annotated_image[:, 0,:,:].mean())
    #     #     print(annotated_image[:, 1, :, :].mean())
    #     #     print(annotated_image[:, 2, :, :].mean())
    #     #     print(torch.Tensor([[annotated_image[:, 0, :, :].mean(),
    #     #                          annotated_image[:, 1, :, :].mean(),
    #     #                          annotated_image[:, 2, :, :].mean()]]).shape)
    #     #     # print([1,1,*annotated_image.shape[-2:]])
    #     #     # A = torch.ones([1, 1, *annotated_image.shape[-2:]])*annotated_image[:, 0, :, :].mean()
    #     #     Mean_img_0 = torch.ones([1, 1, *annotated_image.shape[-2:]])*annotated_image[:, 0, :, :].mean()
    #     #     Mean_img_1 = torch.ones([1, 1, *annotated_image.shape[-2:]])*annotated_image[:, 1, :, :].mean()
    #     #     Mean_img_2 = torch.ones([1, 1, *annotated_image.shape[-2:]])*annotated_image[:, 2, :, :].mean()
    #     #
    #     #     Mean_img = torch.concat((Mean_img_0,Mean_img_1,Mean_img_2), axis = 1)
    #     #
    #     #     ZERO_img = annotated_image.clone()
    #
    #         # print(ZERO_img.shape)
    #         # print(Mean_img.shape)
    #         #
    #         #
    #         # ZERO_img = Mean_img
    #         #
    #         #
    #         # ZERO_annotated = annotated_image.clone()
    #
    #         ZERO_img = torch.zeros([*img.shape])
    #
    #
    #     ZERO_img[:,:,int(x_min):int(x_max),int(y_min):int(y_max)] = crop_img
    #
    #     return ZERO_img.cuda(), None, None #, ZERO_annotated.clone()


    # image_mask = torch.zeros([*img.shape])


    # # ic(image_mask.shape)
    # for obj_id, region in coords_d_ext.items():
    #     # ic(obj_id)
    #     # ic(region)
    #     image_mask[:,:,int(region[2]):int(region[3]),int(region[0]):int(region[1])+1] = 1.0

    # ic(image_mask.shape)

    # new_image_with_crops = img*image_mask.cuda()

    # ic(new_image_with_crops.shape)



def super_pad(img:torch.Tensor, msk:torch.Tensor, nbr_oo: int, hypo=1):
    '''Crops the image, h how much to multiply the Hypotenuse, nbr_oo: number of objects'''
    # discard_the_background, and for every object find the coordinates of the crop
    # ic(msk.shape)
    # msk = msk[1:]
    coords_d = {}
    coords_d_ext = {}
    # ic(img.shape)
    for i, m in enumerate(msk):
        coords = extract_coord_for_crop(m[0])
        coords_d.update({f'{i+1}':coords})
        ext_coords = find_search_region(coords,hypo)
        # ic(coords)
        # ic(ext_coords)
        ext_coords[:2] = torch.clip(ext_coords[:2],0,img.shape[3])
        ext_coords[2:] = torch.clip(ext_coords[2:],0,img.shape[2])
        # ic(ext_coords)
        coords_d_ext.update({f'{i+1}':ext_coords})
        # ic(m.shape)

    # ic(nbr_oo)
    # ic(coords_d_ext)
    # ic(img.shape)
    # ic(msk.shape)
    # pass

    image_mask = torch.zeros([*img.shape])
    # ic(image_mask.shape)
    for obj_id, region in coords_d_ext.items():
        # ic(obj_id)
        # ic(region)
        image_mask[:,:,int(region[2]):int(region[3]),int(region[0]):int(region[1])+1] = 1.0

    # ic(image_mask.shape)

    new_image_with_crops = img*image_mask.cuda()

    # ic(new_image_with_crops.shape)

    return new_image_with_crops.clone()






def crop(img:torch.Tensor, coord:torch.Tensor):
    '''Crop the image. coord -> list [y_min,y_max,x_min, x_max]'''



def extract_coord_for_crop(msk:torch.Tensor) -> torch.Tensor:
    '''Based on the mask, get the coordinates for the crop'''
    msk_non_zeros = msk.nonzero()
    c, r = msk_non_zeros[:,0], msk_non_zeros[:,1]
    # print(c)
    # print(r)
    # print(msk.shape)
    if torch.Tensor([]).size() == c.size():
        x_min = 0.0
        x_max = msk.shape[0]
    else:
        x_min = c.min().item()
        x_max = c.max().item()

    # if r == torch.Tensor([732, 733, 734]):
    #
    #     y_min = r.min().item()
    #     y_min = r.min().item()
    #     print('hi')

    if torch.Tensor([]).size() == r.size():
        y_min = 0.0
        y_max = msk.shape[1]
    else:
        y_min = r.min().item()
        y_max = r.max().item()

    return torch.Tensor([y_min, y_max, x_min, x_max])


def find_search_region(coords:torch.Tensor, hypo_coeff:float) -> torch.Tensor:
    y_min = coords[0]
    x_min = coords[2]
    # print('********')
    # print(coords)
    w = abs(coords[:2].diff()).item()
    h = abs(coords[2:].diff()).item()
    # print(w)
    # print(h)
    ratio_W_wrt_H = w/h if h != 0 else w/1
    hypo = w*w + h*h
    new_hypo = hypo_coeff*hypo
    # ic(hypo)
    # ic(new_hypo)
    # ic(ratio_W_wrt_H)
    new_w = np.sqrt((hypo_coeff*new_hypo*ratio_W_wrt_H**2)/(1+ratio_W_wrt_H**2))
    new_h = np.sqrt((hypo_coeff*new_hypo)/(1+ratio_W_wrt_H**2))
    # ic(w)
    # ic(h)
    # ic(new_w)
    # ic(new_h) ## TODO NE PAS OUBLIER DE RAJOUTER +1 à la fin de la distance pour l'eurreur de l'index

    center_y = w/2+y_min.item()
    center_x = h/2+x_min.item()

    # print(center_y)
    # print(w)
    # print(y_min)

    # ic(center_y)
    # ic(center_x)

    new_y_min = center_y - new_w/2
    new_y_max = center_y + new_w/2
    new_x_min = center_x - new_h/2
    new_x_max = center_x + new_h/2

    # print('###############')
    # print(new_y_min, new_y_max)

    return torch.Tensor([new_y_min, new_y_max, new_x_min, new_x_max]).floor()



