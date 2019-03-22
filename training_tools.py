import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
import random
import torchvision
import os.path
import PIL
from matplotlib import pyplot as plt
import cv2

def binarize(tensor):
    
    binarized = tensor.detach().cpu().numpy()
    std = np.std(binarized)
    binarized = (binarized - np.average(binarized))/std
    binarized = binarized > -0.5

    return torch.tensor(binarized.astype(int))


class LossChecker():
    
    # check_amt is between 0 and 1 -- how much of dataset to check
    def __init__(self, set_type, data_keys, oowl, masks, device, check_amt = 1):
        
        self.batch_size = 1
        self.data_loader = UDataLoader(set_type, data_keys, oowl, masks, self.batch_size, 1, "none")
        self.data_keys = data_keys
        self.device = device
        self.check_amt = check_amt
        self.set_type = set_type
        
    def check_loss(self, model):
        
        self.data_loader.reset()
        
        #loss_function = nn.BCELoss(reduce = False, size_average = False)
        loss_function = nn.BCEWithLogitsLoss(reduce = False, size_average = False)
        avg_loss = 0
        avg_IOU = 0
        losses = []
        
        total_sampled = 0
        #rounds = int(len(self.validation_keys)/self.batch_size)+1
        #for i in range(int((len(self.data_keys)/self.batch_size)*self.check_amt)):
        batch, ground_truth, names, sampled_size = self.data_loader.get_batch()
        #while sampled_size != 0:
        while total_sampled < len(self.data_keys) * self.check_amt and sampled_size !=0:
            with torch.no_grad():

                model.eval()
                total_sampled += sampled_size
                batch = batch.to(device=self.device, dtype=torch.float32)
                ground_truth = ground_truth.to(device=self.device, dtype=torch.float)
                predicted = model(batch)

                loss = loss_function(predicted, ground_truth)
                for x in loss:
                    losses.append(int(torch.sum(x)))
                avg_loss += torch.sum(loss)
                
                predicted_binarized = binarize(predicted)
                IOU = self.get_IOU(predicted_binarized, ground_truth)
                    
                avg_IOU += IOU
                batch, ground_truth, names, sampled_size = self.data_loader.get_batch()
        
        #avg_loss = float(avg_loss) / int(len(self.data_keys)*self.check_amt)
        avg_loss = float(avg_loss) / total_sampled
        
        #avg_IOU = float(avg_IOU) / int(len(self.data_keys)*self.check_amt)
        avg_IOU = float(avg_IOU) / total_sampled

        return avg_loss, avg_IOU


    def get_IOU(self, predicted, gt):
        
        predicted_np = predicted.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()    
        intersection = np.sum(np.logical_and(predicted_np,gt_np))
        union = np.sum(np.logical_or(predicted_np, gt_np))
        return np.float(intersection)/union
        

class UDataLoader:
    
    # data_keys is a list of keys usable in dictionary to load
    def __init__(self, set_type, data_keys, dataset_dict, annotation_dict, batch_size, augment_factor, aug_tricks):
        
        reload_tensors = False
        
        # dataset tensors are 256x256x3 (floattensors)
        # mask tensors are 1x256x256 (floattensors), in range 0/1
        if set_type == "training":
            dataset_pickle_loc = "cache/oowl_train_tensor.pt"
            annotation_pickle_loc = "cache/masks_train_tensor.pt"
        elif set_type == "validation":
            dataset_pickle_loc = "cache/oowl_val_tensor.pt"
            annotation_pickle_loc = "cache/masks_val_tensor.pt"
        elif set_type == "testing":
            dataset_pickle_loc = "cache/oowl_test_tensor.pt"
            annotation_pickle_loc = "cache/masks_test_tensor.pt"
        else:
            raise Exception('unknown set type')

        if not os.path.isfile(dataset_pickle_loc) or not os.path.isfile(annotation_pickle_loc):
            reload_tensors = True
            print("OVERRIDE: reloading tensors; .pt files not found")

        if reload_tensors:

            # loading appropriate data from dictionary
            self.dataset = []
            self.annotations = []


            for key in data_keys:
                self.dataset.append(dataset_dict[key])
                self.annotations.append(annotation_dict[key])
                       
            self.dataset = torch.tensor(self.dataset, dtype = torch.float32)
            torch.save(self.dataset, dataset_pickle_loc)
            self.annotations = torch.tensor(self.annotations, dtype = torch.float32)
            torch.save(self.annotations, annotation_pickle_loc)

        else:

            self.dataset = torch.load(dataset_pickle_loc)
            self.annotations = torch.load(annotation_pickle_loc)
        
        # setting up augmentation tricks
        self.augment_factor = augment_factor

        jitter = T.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1)
        horiz = T.RandomHorizontalFlip()
        crop = T.RandomResizedCrop(256, scale=(0.3,1.0), interpolation = PIL.Image.NEAREST)
        rotate = T.RandomRotation(3, expand = False, resample = PIL.Image.BILINEAR)

        if aug_tricks == 'none':
            self.augmentation_tricks = T.Compose([])
        elif aug_tricks == 'standard':
            self.augmentation_tricks = T.Compose([horiz,crop])
        elif aug_tricks == 'all':
            self.augmentation_tricks = T.Compose([jitter,horiz,crop,rotate])

        self.original_data_idxs = list(range(len(self.dataset)))
        self.available_data_idxs = self.original_data_idxs.copy()
        random.shuffle(self.available_data_idxs)

        self.batch_size = batch_size

        
    def get_batch(self):
        
        batch_keys = self.available_data_idxs[:self.batch_size]
        self.available_data_idxs = self.available_data_idxs[self.batch_size:]
        sampled_size = len(batch_keys)

        if sampled_size == 0:
            return None, None, None, sampled_size

        batch = self.dataset[batch_keys]


        batch = batch.permute(0,3,1,2)
        gt = self.annotations[batch_keys]

        for i in range(self.augment_factor - 1):
            for j in range(sampled_size):
                augmented_img, augmented_mask = self.augment(batch[j],gt[j])
                torch.cat([batch, augmented_img.unsqueeze(0)])
                torch.cat([gt, augmented_mask.unsqueeze(0)])
                
        names = batch_keys
        
        return batch, gt, names, sampled_size

        
    #input: floattensors, img is 3x256x256 in [0,255], mask is 1x256x256 in [0,1] range
    # outputs should be the same dimensions.
    def augment(self, original_tensor_img, original_tensor_mask):
        
        # for ToPILImage to work properly, inputs must be bytetensors of shape 3x256x256, in range [0,255]
        original_tensor_img = original_tensor_img.to(dtype=torch.uint8)
        original_tensor_mask = original_tensor_mask.to(dtype=torch.uint8)
        original_tensor_mask = original_tensor_mask*255
        original_tensor_mask = torch.cat([original_tensor_mask, original_tensor_mask, original_tensor_mask]) 

        seed = random.randint(0,2**32)

        augmented_img = T.ToPILImage()(original_tensor_img)
        random.seed(seed)
        augmented_img = self.augmentation_tricks(augmented_img)
        augmented_img = torchvision.transforms.functional.affine(augmented_img,0,(0,0),1.06,0)
        augmented_img = T.ToTensor()(augmented_img)*255


        # converting to image first by multiplying by 255 and stacking x 3
       
        augmented_mask = T.ToPILImage()(original_tensor_mask)
        random.seed(seed)
        augmented_mask = self.augmentation_tricks(augmented_mask)
        augmented_mask = torchvision.transforms.functional.affine(augmented_mask,0,(0,0),1.06,0)
        augmented_mask = (T.ToTensor()(augmented_mask) > 0.5).type(torch.float)[0]


        return augmented_img, augmented_mask.unsqueeze(0)
        

    
    def reset(self):
        self.available_data_idxs = self.original_data_idxs.copy()
        random.shuffle(self.available_data_idxs)


def show_pic(in_pic):
    in_pic = np.array(in_pic)
    if len(in_pic.shape)== 4:
        in_pic = in_pic.squeeze(0)
    if in_pic.shape[0] == 1:

        orig_pic = in_pic
        pic = np.stack((orig_pic[0], orig_pic[0], orig_pic[0]),0)
        pic = pic*255
        pic = pic.transpose(1,2,0)
    else:
        #pic = cv2.cvtColor(in_pic, cv2.COLOR_BGR2RGB)
        pic = in_pic
        pass
    plt.imshow(pic)
    plt.show()
