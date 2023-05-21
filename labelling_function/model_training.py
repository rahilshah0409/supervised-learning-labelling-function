import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score

def train_model(model, train_data, train_batch_size, test_data, test_batch_size, lr, num_epochs, output_vec_size, events_captured):
    if torch.cuda.is_available():
        model.cuda()

    bce_loss_per_elem = nn.BCEWithLogitsLoss(reduction='sum').cuda() if torch.cuda.is_available() else nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    train_set_size = len(train_data)
    num_batches = math.ceil(train_set_size / train_batch_size)
    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        random.shuffle(train_data)
        for bi in range(num_batches):
            batch = train_data[(bi * train_batch_size) : (bi * train_batch_size) + train_batch_size]
            batch_input, batch_target = zip(*batch)
            batch_target = convert_events_to_output_vectors(batch_target, output_vec_size, events_captured)
            batch_input = Variable(torch.FloatTensor(np.array(batch_input)), requires_grad=True)
            batch_target = Variable(torch.FloatTensor(np.array(batch_target)))

            if torch.cuda.is_available():
                batch_target, batch_input = batch_target.cuda(), batch_input.cuda()

            batch_output = model.forward(batch_input)

            loss = bce_loss_per_elem(batch_output, batch_target)

            optimizer.zero_grad()
            total_train_loss += loss.item()
            loss.backward()

            # Don't know what the below line does, look into
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        avg_train_loss = round(total_train_loss / num_batches, 5)
        avg_test_loss, precision, recall, f1 = eval_model(model, test_data, test_batch_size, events_captured, output_vec_size)
        avg_test_loss = round(avg_test_loss, 5)
        epoch_train_losses.append(avg_train_loss)
        epoch_test_losses.append(avg_test_loss)
        wandb.log({"epoch": epoch, 
                   "train_loss": avg_train_loss, 
                   "test_loss": avg_test_loss, 
                   "test_precision": precision, 
                   "test_recall": recall, 
                   "test_f1": f1})

        print("Epoch: {}. Training loss: {}. Test loss: {}".format(epoch, avg_train_loss, avg_test_loss))

    return model

def eval_model(model, test_data, batch_size, events_captured, output_vec_size):
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    num_batches = math.ceil(len(test_data) / batch_size)
    total_loss = 0
    acc = []
    pred = []
    with torch.no_grad():
        for bi in range(num_batches):
            batch = test_data[(bi * batch_size) : (bi * batch_size) + batch_size]

            batch_input, batch_target = zip(*batch)
            batch_target = convert_events_to_output_vectors(batch_target, output_vec_size, events_captured)
            batch_input = Variable(torch.FloatTensor(np.array(batch_input)), requires_grad=True)
            batch_target = Variable(torch.FloatTensor(np.array(batch_target)))

            if torch.cuda.is_available():
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()

            batch_output = model(batch_input)
            acc.extend(batch_target.cpu().numpy())
            pred.extend(torch.sigmoid(batch_output).cpu().numpy())

            # print("Batch output and label in dataset")
            # wrong_predictions = [(torch.round(torch.sigmoid(output)), target) for output, target in zip(batch_output, batch_target) if not torch.equal(torch.round(torch.sigmoid(output)), target)]
            # print(wrong_predictions)

            bce_loss_per_elem = nn.BCEWithLogitsLoss(reduction='sum').cuda() if torch.cuda.is_available() else nn.BCEWithLogitsLoss(reduction='sum')
            loss = bce_loss_per_elem(batch_output, batch_target)
            total_loss += loss.item()
            # print("Batch: {}. Loss on test set: {}".format(bi, loss.item()))

    # Return average loss per batch
    avg_loss = total_loss / num_batches
    acc = np.concatenate(acc, axis=0)
    pred = np.concatenate(pred, axis=0)
    pred_binary = (pred >= 0.5).astype(int)

    # We calculate the precision, recall and f1 score via micro averaging since the dataset is imbalanced and each class is treated equally despite imbalance
    precision = precision_score(acc, pred_binary, average='micro')
    recall = recall_score(acc, pred_binary, average='micro')
    f1 = f1_score(acc, pred_binary, average='micro')

    return avg_loss, precision, recall, f1

def convert_events_to_output_vectors(events_list, output_vec_size, events_captured):
    vectors_list = []
    events_captured_list = sorted(list(filter(lambda pair: pair[0] == "black" or pair[1] == "black", events_captured)))
    for i in range(len(events_list)):
        events = events_list[i]
        output_vector = np.zeros(output_vec_size)
        for event in events:
            # if event[0] > event[1]:
            #     event = (event[1], event[0])
            if (event in events_captured_list):
                element_i = events_captured_list.index(event)
                output_vector[element_i] = 1
        vectors_list.append(output_vector)
    return vectors_list