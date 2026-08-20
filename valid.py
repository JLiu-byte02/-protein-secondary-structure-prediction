import os
import numpy as np
import torch

seq_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M',
            'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', '']

def acc(pred_list, label_list, ignore_cls=8):
    total = 0
    correct = 0

    edge_total = 0
    edge_correct = 0
    for pred_sub, label_sub in zip(pred_list, label_list):
        for idx, (p, l) in enumerate(zip(pred_sub, label_sub)):
            if l != torch.tensor(ignore_cls).to('cuda'):
                if idx == 0 or idx == len(label_sub)-1:
                    if p == l:
                        edge_correct += 1
                    edge_total += 1
                elif l != label_sub[idx-1] or l != label_sub[idx+1]:
                    if p == l:
                        edge_correct += 1
                    edge_total += 1

                if p == l:
                    correct += 1
                total += 1
    return correct / total, edge_correct / edge_total

def get_valid_acc(config, model, valid_loader, epoch, test_mark, save_acc=True):
    model.eval()
    
    count = 0
    acc_valid_sum = 0
    edge_acc_sum = 0
    dict_epoch = {}

    value_acc_list = []
    value_edge_acc_list = []

    for x, labels in valid_loader:
        outputs_val = model(x.to('cuda'))
        prediction = torch.argmax(outputs_val, 1)
        value_acc, value_edge_acc = acc(prediction, labels.to('cuda'), ignore_cls=config.class_num-1)
        value_acc_list.append(value_acc)
        value_edge_acc_list.append(value_edge_acc)
        acc_valid_sum += value_acc
        edge_acc_sum += value_edge_acc
        count += 1

        if save_acc:
            key_x = x[0].permute(1, 0)
            dict_epoch[key_x] = value_acc  

    if save_acc:
        save_acc_path = os.path.join(config.save_acc_fpath,
                                    'acc_{}_Epoch{}_{}.acc'.format(test_mark, epoch + 1, config.model))

        with open(save_acc_path, 'w') as w:
            for key in dict_epoch:
                seq = ''
                for line in key:
                    for idx, item in enumerate(line[:21]):
                        if item == 1:
                            char = seq_list[idx]
                            seq += char
                            char = ''
                            break
                w.write('{},{}\n'.format(seq, dict_epoch[key]))
                seq = ''

    # return acc_valid_sum / count, edge_acc_sum / count
    return np.mean(value_acc_list), np.mean(value_edge_acc_list), np.std(value_acc_list), np.std(value_edge_acc_list)