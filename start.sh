# name, device, epochs, batch size, augument, depth, relu, dropout, lr, optim, aug_tricks
python3 batch_run.py no_aug 7 60 16 8 5 relu 0.2 0.001 adam none
#python3 batch_run.py fixed_val_dropout_fixed_aug_save_each 7 60 16 8 5 relu 0.2 0.001 adam all
#python3 batch_run.py new_dropout 4 1 16 2 5 relu 0.2 0.001 adam
