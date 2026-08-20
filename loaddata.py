import random
import torch

def read_data(path, config, shuffle=False):
    feature_list = []
    label_list = []
    with open(path) as f:
        feature_tmp = []
        label_tmp = []
        for index, line in enumerate(f):
            ll = line.split(' ')
            if len(ll) > 1:
                ll = list(map(float, ll[:-1]))
                if len(ll) != config.feature_size:
                    raise selfException("the length of ll is not {}".format(config.feature_size))
                feature_tmp.append(ll)
            elif(int(float(ll[0].strip())) < 9):
                label_tmp.append(int(float(ll[0].strip())))
            if len(feature_tmp) == config.data_max_text_len:
                feature_list.append(feature_tmp)
                feature_tmp = []
            if len(label_tmp) == config.data_max_text_len:
                label_list.append(label_tmp)
                label_tmp = []
        if shuffle:
            randnum = random.randint(0,100)
            random.seed(randnum)
            random.shuffle(feature_list)
            random.seed(randnum)
            random.shuffle(label_list)
            print('shuffled random seed: {}'.format(randnum))
    return torch.tensor(feature_list), torch.tensor(label_list)

# def read_data(path, config):
#     feature_list = []
#     label_list = []
#     with open(path) as f:
#         feature_tmp = []
#         label_tmp = []
#         for index, line in enumerate(f):
#             ll = line.split(' ')
#             if len(ll) > 1:
#                 ll = list(map(float, ll[:-1]))
#                 if len(ll) != 44:
#                     raise selfException("the length of ll is not {}".format(44))
#                 feature_tmp.append(ll)
#             elif(int(float(ll[0].strip())) < 9):
#                 label_tmp.append(int(float(ll[0].strip())))
#             if len(feature_tmp) == config.data_max_text_len:
#                 feature_list.append(feature_tmp)
#                 feature_tmp = []
#             if len(label_tmp) == config.data_max_text_len:
#                 label_list.append(label_tmp)
#                 label_tmp = []
#         randnum = random.randint(0,100)
#         random.seed(randnum)
#         random.shuffle(feature_list)
#         random.seed(randnum)
#         random.shuffle(label_list)
#         print('shuffled random seed: {}'.format(randnum))
#     return feature_list, label_list

# def dataset_splits(feature_list, label_list, splits=['train', 'valid']):
#     feature_dict = {}
#     label_dict = {}
#     for split in splits:
#         if split == 'train':
#             feature_dict[split] = torch.tensor(feature_list[:int(len(feature_list)*9/10)]) #, dtype=torch.long , dtype=torch.double
#             label_dict[split] = torch.tensor(label_list[:int(len(feature_list)*9/10)])
#         elif split == 'valid':
#             feature_dict[split] = torch.tensor(feature_list[int(len(feature_list)*9/10):])
#             label_dict[split] = torch.tensor(label_list[int(len(feature_list)*9/10):])
#     return feature_dict, label_dict