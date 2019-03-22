import numpy as np
import os
import sys
import cv2


def get_data(img_path, mask_path):
    
    masks = {}
    oowl = {}

    for root, dir, files in os.walk(mask_path):
        for file in files:
            filename = root + '/' + file
            if filename[-3:] == 'png':
                mask_img = cv2.imread(filename,1)




    return oowl, masks, ids

def process_masks(mask_path, new_dims_mask):

    # 132 for non-padded version

    enable_trigger = False
    trigger = False
    masks = {}
    i = 0
    for root, dir, files in os.walk(mask_path):
        if (enable_trigger and not trigger) or not enable_trigger:
            for file in files:
                filename = root + '/' + file
                if filename[-3:] == 'png':
                    mask_img = cv2.imread(filename,1)
                    if new_dims_mask != mask_img.shape[1]:
                        mask_img = cv2.resize(mask_img, (new_dims_mask, new_dims_mask), interpolation=cv2.INTER_NEAREST)
                    #mask_img = torch.tensor(mask_img[:,:,1], dtype = torch.float32) / 255
                    mask_img = (np.array([mask_img[:,:,1]])/ 255).tolist()

                    # adding stripped version of pic file directory as key
                    pic_id = filename[:-4].split('/')[1:]
                    pic_id[-1] = pic_id[-1].replace("Mask_","")
                    pic_id = '/'.join(pic_id)
                    masks[pic_id] = mask_img
                    trigger = True

                    i+=1
                    if i %500 ==0:
                        print(i)
        else:
            break

    return masks


def process_oowl(img_path, new_dims_oowl, masks):

    # 316 for non-padded version

    oowl = {}
    i=0
    enable_trigger = False
    trigger = False

    for root, dir, files in os.walk(img_path):
        if (enable_trigger and not trigger) or not enable_trigger:
            for file in files:
                pic_id = (root+'/'+file).replace(img_path, '')[1:-4]
                if pic_id in masks:
                    pic_path = root+'/'+file
                    oowl_img = cv2.imread(pic_path,1)
                    if new_dims_oowl != oowl_img.shape[1]:
                        oowl_img = cv2.resize(oowl_img, (new_dims_oowl,new_dims_oowl))
                    #oowl_img = torch.tensor(oowl_img, dtype = torch.float32)
                    oowl[pic_id] = oowl_img
                    
                    trigger = True

                    i+=1
                    if i %500==0:
                        print(i)
        else:
            break

    return oowl
