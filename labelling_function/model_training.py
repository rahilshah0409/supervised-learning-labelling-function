import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

def train_model(model, lr, num_epochs, train_data, batch_size, output_vec_size, events_captured):
    bceloss = nn.BCELoss().cuda() if torch.cuda.is_available() else nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    train_set_size = len(train_data)
    num_batches = math.ceil(train_set_size / batch_size)
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        random.shuffle(train_data)
        for bi in range(num_batches):
            batch = train_data[(bi * batch_size) : (bi * batch_size) + batch_size]
            batch_input, batch_target = zip(*batch)
            batch_target = convert_events_to_output_vectors(batch_target, output_vec_size, events_captured)
            batch_output = model.forward(batch_input)

            optimizer.zero_grad()
            loss = bceloss(batch_output, batch_target)
            total_train_loss += loss.item()
            loss.backward()
            # Don't know what the below line does, look into
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        avg_loss = round(total_train_loss / num_batches, 5)
        epoch_losses.append(avg_loss)
        wandb.log({"epoch": epoch, "loss": avg_loss})

        # Maybe get the loss when tested against unseen test data. Need to make sure that the test data is truly unseen

        print("Epoch: {}. Training loss: {}".format(epoch, avg_loss))
    
    # TODO: plot results of training across epochs here

def eval_model(model, test_data, batch_size, events_captured, output_vec_size):
    model.eval()
    num_batches = math.ceil(len(test_data) / batch_size)
    total_loss = 0
    with torch.no_grad():
        for bi in range(num_batches):
            batch = test_data[(bi * batch_size) : (bi * batch_size) + batch_size]

            batch_input, batch_target = zip(*batch)
            batch_target = convert_events_to_output_vectors(batch_target, output_vec_size, events_captured)

            if torch.cuda.is_available():
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()

            batch_output = model(batch_input)

            bce_loss = nn.BCELoss.cuda() if torch.cuda.is_available() else nn.BCELoss
            loss = bce_loss(batch_output, batch_target)
            total_loss += loss.item()

    # Return average loss per batch
    return total_loss / num_batches

# TODO: Implement this function
def convert_events_to_output_vectors(events_list, output_vec_size, events_captured):
    vectors_list = []
    events_captured_list = sorted(list(events_captured))
    for i in range(len(events_list)):
        events = events_list[i]
        output_vector = np.zeros(output_vec_size)
        for event in events:
            element_i = 0 # It won't actually be 0 but find a way to do this in a sensible way
            vectors_list[element_i] = 1
        vectors_list.append(output_vector)
    return vectors_list