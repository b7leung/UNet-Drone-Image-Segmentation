import torch.optim as optim
import torch.nn as nn
import torch
from datetime import *
from dataset_processing import *
from UNet import *
from training_tools import *
import time
import random


class Trainer():

    def __init__(self, model, hyperparameters, training_keys, validation_keys, training_dict, annotation_dict):
        
        self.model_name = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')+"_"+hyperparameters['name']+"_"+str(random.randint(1,1000000))
        self.log = open("models/"+self.model_name+".txt",'w')
        for key in hyperparameters:
            self.record(key+": "+str(hyperparameters[key]))

        self.device = hyperparameters['device']
        self.model = model
        self.model.to(device = self.device)

        self.training_keys = training_keys
        self.validation_keys = validation_keys
        self.training_dict = training_dict
        self.annotation_dict = annotation_dict
        
        self.print_every = 25
        self.batch_size = hyperparameters['batch_size']
        self.epochs = hyperparameters['epochs']
        self.augument_factor = hyperparameters['augument_factor']
        self.aug_tricks = hyperparameters['aug_tricks']
        self.learning_rate = hyperparameters['learning_rate']
        self.optimizer = hyperparameters['optimizer']
        self.save_each = False

    def train(self):

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0)
        elif self.optimizer == 'sgd':
            optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0)
        else:
            raise Exception('optimizer not recognized; must be adam or sgd')

        loss_function = nn.BCEWithLogitsLoss()
        dataLoader = UDataLoader('training',self.training_keys, self.training_dict, self.annotation_dict, self.batch_size, self.augument_factor, self.aug_tricks)

        validation_loss_checker = LossChecker("validation",self.validation_keys, self.training_dict, self.annotation_dict, self.device)
        training_loss_checker = LossChecker("training",self.training_keys, self.training_dict, self.annotation_dict, self.device, check_amt = 0.1)
        
        start_time = time.time()

        highest_avg_IOU = 0
        lowest_validation_loss = None

        for epoch in range(self.epochs):

            self.record("Epoch " + str(epoch) + " .............................................")
            self.record("time elapsed: " + str(time.strftime("%H:%M:%S", time.gmtime(time.time()- start_time))))

            total_sampled = 0
            
            batch, ground_truth, names, sampled_size = dataLoader.get_batch()
            while sampled_size != 0:
            #for batch_num in range(int(len(self.training_keys)/self.batch_size)+1):
                
                total_sampled += sampled_size
                
                self.model.train()

                batch = batch.to(device=self.device, dtype=torch.float)
                ground_truth= ground_truth.to(device=self.device, dtype=torch.float)

                predicted_masks = self.model(batch)

                loss = loss_function(predicted_masks, ground_truth)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                
                
                curr_progress = float(total_sampled)/len(self.training_keys)

                if int(curr_progress*100) % self.print_every == 0:
                    self.record('>> Percent done: ' + str(curr_progress))

                batch, ground_truth, names, sampled_size = dataLoader.get_batch()

            epoch_validation_loss, epoch_avg_validation_IOU = validation_loss_checker.check_loss(self.model)
            epoch_training_loss, epoch_avg_training_IOU = training_loss_checker.check_loss(self.model)
            self.record("Epoch Validation Loss: " + str(epoch_validation_loss))
            self.record("Epoch Validation Avg IOU: " + str(epoch_avg_validation_IOU))
            self.record("Epoch Training Loss: " + str(epoch_training_loss))
            self.record("Epoch Training Avg IOU: " + str(epoch_avg_training_IOU))

            if epoch_avg_validation_IOU > highest_avg_IOU:
                highest_avg_IOU = epoch_avg_validation_IOU
                if self.save_each:
                    torch.save(self.model, "models/"+self.model_name + "_highest_avg_iou_"+str(epoch))
                else:
                    torch.save(self.model, "models/"+self.model_name + "_highest_avg_iou")
                    

            if lowest_validation_loss == None or epoch_validation_loss < lowest_validation_loss: 
                lowest_validation_loss = epoch_validation_loss
                torch.save(self.model, "models/"+self.model_name + "_lowest_loss")

            
            dataLoader.reset()

        end_time = time.time()
        self.record("time elapsed: " + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))


    def record(self, string):
        self.log = open("models/"+self.model_name+".txt",'a')
        print(string)
        self.log.write(string+'\n')
        self.log.close()


        
