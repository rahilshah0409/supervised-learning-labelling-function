import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Trains the model with a training dataset
def train_model(model, train_data, train_batch_size, test_data, test_batch_size, lr, num_epochs, output_vec_size, events_captured):
    if torch.cuda.is_available():
        model.cuda()

    bce_loss_per_elem = nn.BCEWithLogitsLoss(reduction='mean').cuda() if torch.cuda.is_available() else nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    train_set_size = len(train_data)
    num_batches = math.ceil(train_set_size / train_batch_size)

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

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        avg_train_loss = round(total_train_loss / num_batches, 5)
        avg_test_loss, precision_scores, recall_scores, f1_scores, accuracy_scores = eval_model(model, test_data, test_batch_size, events_captured, output_vec_size)
        avg_test_loss = round(avg_test_loss, 5)
        
        wandb.log({"epoch": epoch, 
                   "train_loss": avg_train_loss, 
                   "test_loss": avg_test_loss})
        for event in precision_scores.keys():
            wandb.log({"epoch": epoch,
                       str(event) + "_test_precision": precision_scores[event],
                       str(event) + "_test_recall": recall_scores[event],
                       str(event) + "_test_f1": f1_scores[event],
                       str(event) + "_test_accuracy": accuracy_scores[event]})
            

        print("Epoch: {}. Training loss: {}. Test loss: {}".format(epoch, avg_train_loss, avg_test_loss))

    return model, precision_scores, recall_scores, f1_scores, accuracy_scores

# Evaluates the model, calculating the accuracy, precision, recall and F1 for each class
def eval_model(model, test_data, batch_size, events_captured, output_vec_size):
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    num_batches = math.ceil(len(test_data) / batch_size)
    total_loss = 0
    acc = []
    pred = []
    events_captured.append("no_event")
    precision_scores = {event: 0.0 for event in events_captured}
    recall_scores = {event: 0.0 for event in events_captured}
    f1_scores = {event: 0.0 for event in events_captured}
    accuracy_scores = {event: 0.0 for event in events_captured}

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
            acc.append(batch_target.cpu().numpy())
            pred.append(torch.sigmoid(batch_output).cpu().numpy())

            bce_loss_per_elem = nn.BCEWithLogitsLoss(reduction='mean').cuda() if torch.cuda.is_available() else nn.BCEWithLogitsLoss(reduction='mean')
            loss = bce_loss_per_elem(batch_output, batch_target)
            total_loss += loss.item()

    acc = np.concatenate(acc, axis=0)
    pred = np.concatenate(pred, axis=0)
    pred_binary = (pred >= 0.5).astype(int)

    for event_i in range(output_vec_size):
        precision = precision_score(acc[:, event_i], pred_binary[:, event_i], zero_division='warn')
        recall = recall_score(acc[:, event_i], pred_binary[:, event_i], zero_division='warn')
        f1 = f1_score(acc[:, event_i], pred_binary[:, event_i], zero_division='warn')
        accuracy = accuracy_score(acc[:, event_i], pred_binary[:, event_i])
        event_tuple = events_captured[event_i]
        precision_scores[event_tuple] = precision
        recall_scores[event_tuple] = recall
        f1_scores[event_tuple] = f1
        accuracy_scores[event_tuple] = accuracy

    no_event_precision = precision_score(np.sum(acc, axis=1) == 0, np.sum(pred_binary, axis=1) == 0, zero_division=0)
    no_event_recall = recall_score(np.sum(acc, axis=1) == 0, np.sum(pred_binary, axis=1) == 0, zero_division=0)
    no_event_f1 = f1_score(np.sum(acc, axis=1) == 0, np.sum(pred_binary, axis=1) == 0, zero_division=0)
    no_event_accuracy = accuracy_score(np.sum(acc, axis=1) == 0, np.sum(pred_binary, axis=1) == 0)

    precision_scores["no_event"] = no_event_precision
    recall_scores["no_event"] = no_event_recall
    f1_scores["no_event"] = no_event_f1
    accuracy_scores["no_event"] = no_event_accuracy
    
    # Return average loss per batch
    avg_loss = total_loss / num_batches

    return avg_loss, precision_scores, recall_scores, f1_scores, accuracy_scores

# Converts events into output vectors depending on the order of events in the event vocab
def convert_events_to_output_vectors(events_list, output_vec_size, events_captured):
    vectors_list = []
    events_captured_list = sorted(list(filter(lambda pair: pair[0] == "black" or pair[1] == "black", events_captured)))
    for i in range(len(events_list)):
        events = events_list[i]
        output_vector = np.zeros(output_vec_size)
        for event in events:
            if (event in events_captured_list):
                element_i = events_captured_list.index(event)
                output_vector[element_i] = 1
        vectors_list.append(output_vector)
    return vectors_list

# Given an output vector, this method returns the corresponding event set.
def convert_output_vectors_to_events(output_vector_list, output_vec_size, events_captured):
    events_list = []
    for output_vec in output_vector_list:
        events = set()
        for i in range(output_vec_size):
            if output_vec[i] == 1.0:
                events.add(events_captured[i])
        events_list.append(events)
    return events_list