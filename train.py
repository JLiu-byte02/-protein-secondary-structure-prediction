import torch
import torch.nn as nn
from torchcrf import CRF
import torch.utils.data as Data
import torch.nn.functional as F

from torch.autograd import Variable
import torch.utils.data as data
import configparser
import os
import numpy as np
import random

import model_condgated_cbe_aspp
import valid

import loaddata
import time
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

import smile as sm
from smile import flags, logging

# flags.DEFINE_float("lr",0.0001," ")
# flags.DEFINE_integer("cnn_layer_num",3," ")

flags.DEFINE_string(
    "data_fpath",
    r"/root",
    " ")
flags.DEFINE_string("train_fname", "train.feat", " ")
flags.DEFINE_string("valid_fname", "valid.feat", " ")
flags.DEFINE_string("test_fname", "test.feat", " ")
flags.DEFINE_string("test_mark", "cb513", " ")
flags.DEFINE_integer("num_gpus", 2, "number of GPUs to use")
flags.DEFINE_boolean("load_model", False, " ")
flags.DEFINE_boolean("reset_optimizer", True, " ")
flags.DEFINE_boolean("save_acc", True, " ")
flags.DEFINE_integer("batch_size", 32, " ")
flags.DEFINE_string("label", "ss", "ss/rsa")


flags.DEFINE_string("model_path", "models/", " ")
flags.DEFINE_float("lr", 0.0001, " ")
flags.DEFINE_string("MSlr_milestones", "10,30", " ")
flags.DEFINE_string("activationFunc", "default", "logsigmoid/sigmoid/tanh/none")
flags.DEFINE_integer("num_epochs", 100, " ")

flags.DEFINE_integer("fc_hidden_size", 800, " ")

flags.DEFINE_integer("lstm_hidden_size", 512, " ")
flags.DEFINE_integer("blstm_layer_num", 2, " ")
flags.DEFINE_float("lstm_dropout_rate", 0.0, "0.5")  # aclstm
flags.DEFINE_float("fc1_dropout_rate", 0.0, "0.4")  # aclstm

flags.DEFINE_float("fc0_dropout_rate", 0.0, "0.5")  # aclstm

flags.DEFINE_string("lr_sche", "RlrOP", "MSlr/RlrOP")

# flags.DEFINE_boolean("save_step",False," ")

flags.DEFINE_string("loss_function", "cel", "cel(CrossEntropyLoss)/crf")
flags.DEFINE_string(
    "model", "condgatedcbeaspp",
    "gatedcnn/dl/cnnaspp/lstmaspp/lstm/aclstm/aclstmaspp/aclstmaspp02/informer/condgatedcbeaspp")

flags.DEFINE_boolean("resnet", True, " ")
flags.DEFINE_boolean("conv_end", True, " ")

flags.DEFINE_float("dropout", 0.0, "0.5")
flags.DEFINE_integer("n_layers", 64, " ")
flags.DEFINE_integer("kernel_width", 3, " ")
flags.DEFINE_integer("out_channel", 64, " ")
flags.DEFINE_integer("res_block_cnt", 2, " ")
flags.DEFINE_float("cond_dropout_rate", 0.2, "0.0, 0.2")


flags.DEFINE_integer("d_model", 256, " ")
flags.DEFINE_integer("e_layers", 3, " ")
flags.DEFINE_integer("n_heads", 8, " ")
flags.DEFINE_string("attn", "reg", "prob/reg")

flags.DEFINE_integer("cnn_layer_num", 3, " ")
flags.DEFINE_integer("bot_layer_num", 1, " ")
flags.DEFINE_string("window_size_list", "1,3,3,3", "")
flags.DEFINE_string("dilation_list", "1,2,4,8", "")
flags.DEFINE_float("output_dropout_rate", 0.0, "0.0, 0.2")
flags.DEFINE_string("output_layer", "conv1", "conv1/convtop/fc")
flags.DEFINE_string("mid_layer", "fc", "conv1/fc")

flags.DEFINE_integer("node_size", 100, " ")

FLAGS = flags.FLAGS


class Config:


    data_fpath = FLAGS.data_fpath
    train_fname = FLAGS.train_fname
    valid_fname = FLAGS.valid_fname
    test_fname_list = FLAGS.test_fname

    test_fname_list = test_fname_list.split(',')

    train_fpath = os.path.join(data_fpath, train_fname)
    valid_fpath = os.path.join(data_fpath, valid_fname)
    test_fpath_list = [(os.path.join(FLAGS.data_fpath, fname)) for fname in test_fname_list]

    test_mark_list = FLAGS.test_mark
    test_mark_list = test_mark_list.split(',')

    load_model = FLAGS.load_model
    reset_optimizer = FLAGS.reset_optimizer

    save_acc = FLAGS.save_acc
    save_acc_fpath = FLAGS.model_path



    device = 'cuda'
    # tmp_w_size = 11

    dropout_rate = 0.0
    output_dropout_rate = FLAGS.output_dropout_rate
    # embedding_size = 256
    feature_size = 42
    node_size = FLAGS.node_size
    data_max_text_len = 700
    cnn_layer_num = FLAGS.cnn_layer_num

    # bot_window_size = tmp_w_size
    # window_sizes = [tmp_w_size] * int(cnn_layer_num - 2)
    top_window_size = tmp_w_size
    is_train = True
    batch_size = FLAGS.batch_size

    lstm_hidden_size = FLAGS.lstm_hidden_size
    blstm_layer_num = FLAGS.blstm_layer_num
    lstm_dropout_rate = FLAGS.lstm_dropout_rate
    fc1_dropout_rate = FLAGS.fc1_dropout_rate

    fc0_dropout_rate = FLAGS.fc0_dropout_rate

    model = FLAGS.model
    activationFunc = FLAGS.activationFunc
    loss_function = FLAGS.loss_function

    model_path = FLAGS.model_path
    if FLAGS.model_path[-1] == '/':
        log_path = '{}.log'.format(FLAGS.model_path[:-1])
    else:
        log_path = '{}.log'.format(FLAGS.model_path)

    if FLAGS.label == 'ss':
        class_num = 9
    # elif FLAGS.label == 'rsa':
    #     class_num = 4
    lr = FLAGS.lr
    weight_decay = 1e-5
    min_lr = 0.000000001
    num_epochs = FLAGS.num_epochs
    lr_sche = FLAGS.lr_sche
    num_gpus = FLAGS.num_gpus

    MSlr_milestones = [int(i) for i in FLAGS.MSlr_milestones.split(',')]

    step_jump_output = 5

    # fc:
    # hidden_size = FLAGS.fc_hidden_size
    # window_size = 10


    output_layer = FLAGS.output_layer

    dilation_list = [int(i) for i in FLAGS.dilation_list.split(',')]
    window_size_list = [int(i) for i in FLAGS.window_size_list.split(',')]

    if model == 'condgatedcbeaspp':
        # data config
        MAX_SENT = 100
        embed_dim = 42
        vocab_size = 9

        #GCNN config
        n_layers = FLAGS.n_layers
        kernel_width = FLAGS.kernel_width
        out_channel = FLAGS.out_channel
        res_block_cnt = FLAGS.res_block_cnt 
        dropout = FLAGS.dropout

        conv_end = FLAGS.conv_end
        resnet = FLAGS.resnet

        cond_dropout_rate = FLAGS.cond_dropout_rate

    


if __name__ == '__main__':
    start = datetime.datetime.now()

    config = Config()
    print("Model Structure Parameter: ", Config.__dict__)
    print('================================')
    start = datetime.datetime.now()
    write_log = open(config.log_path, 'w')

    write_log.write("Model Structure Parameter: ")
    for key in Config.__dict__:
        write_log.write('{}:{}, '.format(key, Config.__dict__[key]))
    # write_log.write(Config.__dict__)
    write_log.write('\n')
    write_log.write('================================\n')
    write_log.flush()




    def read_data(path):
        feature_list = []
        label_list = []
        with open(path) as f:
            feature_tmp = []
            label_tmp = []
            for index, line in enumerate(f):
                ll = line.split(' ')
                if len(ll) > 1:
                    ll = list(map(float, ll[:-1]))
                    if len(ll) != 44:
                        raise selfException(
                            "the length of ll is not {}".format(44))
                    feature_tmp.append(ll)
                elif (int(float(ll[0].strip())) < 9):
                    label_tmp.append(int(float(ll[0].strip())))
                if len(feature_tmp) == config.data_max_text_len:
                    feature_list.append(feature_tmp)
                    feature_tmp = []
                if len(label_tmp) == config.data_max_text_len:
                    label_list.append(label_tmp)
                    label_tmp = []
            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(feature_list)
            random.seed(randnum)
            random.shuffle(label_list)
            print('shuffled random seed: {}'.format(randnum))
        return feature_list, label_list


    # read data
    train_data, train_label = loaddata.read_data(
        config.train_fpath, config, shuffle=True)
    train_data = train_data.permute(0, 2, 1)
    valid_data, valid_label = loaddata.read_data(
        config.valid_fpath, config, shuffle=False)
    valid_data = valid_data.permute(0, 2, 1)

    test_data_list = []
    test_label_list = []
    for test_path in config.test_fpath_list:
        test_data, test_label = loaddata.read_data(
            test_path, config, shuffle=False)
        test_data = test_data.permute(0, 2, 1)
        test_data_list.append(test_data)
        test_label_list.append(test_label)


    print('train_data_size', train_data.size())
    print('train_label_size', train_label.size())

    if config.model == 'condgatedcbeaspp':
        CNNCRFModel = model_condgated_cbe_aspp.GatedCBEASPP(
            config.vocab_size,
            config.embed_dim,
            config.kernel_width,
            config.out_channel,
            config.n_layers,
            config.res_block_cnt,
            config.activationFunc,
            config,
            dropout=config.dropout)

        
    print(CNNCRFModel)

    if torch.cuda.is_available():
        print('cuda is available')
        # 根据 flags.num_gpus 使用多卡并行训练（保持原有打印不变）
        device_ids = list(range(config.num_gpus)) if config.num_gpus > 0 else []
        CNNCRFModel = CNNCRFModel.cuda()
        if len(device_ids) > 1:
            CNNCRFModel = nn.DataParallel(CNNCRFModel, device_ids=device_ids)
        crf = CRF(config.class_num, batch_first=True).cuda()
        # crf = nn.DataParallel(crf)

    # loss_func = nn.CrossEntropyLoss()
    if config.class_num == 9:
        loss_func = nn.CrossEntropyLoss(ignore_index=8)
    elif config.class_num == 4:
        loss_func = nn.CrossEntropyLoss(ignore_index=3)

    torch_dataset = Data.TensorDataset(train_data, train_label)
    # put dataset in DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=config.batch_size,  # mini batch size
        shuffle=True,
        num_workers=2,
    )

    valid_torch_dataset = Data.TensorDataset(valid_data, valid_label)
    valid_loader = Data.DataLoader(
        dataset=valid_torch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    test_loader_list = []
    for test_data, test_label in zip(test_data_list, test_label_list):
        test_torch_dataset = Data.TensorDataset(test_data, test_label)
        test_loader = Data.DataLoader(
            dataset=test_torch_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )
        test_loader_list.append(test_loader)


    optimizer = torch.optim.Adam(
        CNNCRFModel.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if FLAGS.lr_sche == 'RlrOP':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=3,
            factor=0.1,
            min_lr=config.min_lr,
            verbose=True)
    elif config.lr_sche == 'MSlr':
        lr_scheduler = MultiStepLR(optimizer, config.MSlr_milestones)


    def acc(pred_list, label_list):
        total = 0
        good = 0
        for pred_sub, label_sub in zip(pred_list, label_list):
            for p, l in zip(pred_sub, label_sub):
                if l != torch.tensor(config.class_num-1).to('cuda'):
                    if p == l:
                        good += 1
                    total += 1
        return good / total


    def tr_va_acc(outputs, label, acc_1):
        prediction = torch.argmax(outputs, 1)
        acc_1 += acc(prediction, label)
        # acc_1 = acc_1 / (step+1)
        return acc_1

    start_epoch = 0
    if os.path.exists(os.path.join(config.model_path, 'Epoch_last.ckpt')) and config.load_model:
        checkpoint = torch.load(os.path.join(config.model_path, 'Epoch_last.ckpt'))
        # 兼容 DataParallel 与单卡的加载
        if hasattr(CNNCRFModel, 'module'):
            CNNCRFModel.module.load_state_dict(checkpoint['model'])
        else:
            CNNCRFModel.load_state_dict(checkpoint['model'])
        if not config.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    best_epoch = 0
    best_valid_acc = 0
    for epoch in range(start_epoch, config.num_epochs):
        CNNCRFModel.train()
        acc_1 = 0
        tt_1 = 0
        loss_epoch_sum = 0


        for step, (batch_x, batch_y) in enumerate(loader):
            # start training
            optimizer.zero_grad()
            outputs = CNNCRFModel(batch_x.to('cuda'))


            # if config.loss_function == 'crf':
            #     outputs = outputs.permute(0, 2, 1)
            #     mask = (batch_y.to('cuda') != 8).long().to(torch.uint8)
            #     loss = -crf(outputs, batch_y.to('cuda'), mask=mask)
            #     outputs_acc = outputs.permute(0, 2, 1)
            if config.loss_function == 'cel':
                loss = loss_func(outputs, batch_y.to('cuda'))
                outputs_acc = outputs

            loss.backward()
            optimizer.step()
            loss_epoch_sum += loss

            acc_1 = tr_va_acc(outputs_acc, batch_y.to('cuda'), acc_1)
            tt_1 += 1

     
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)

        loss_avg = loss_epoch_sum / (step+1)
        print_line = 'Epoch [{}/{}], Loss: {:.5f}, train_acc: {:.5f}, '.format(epoch + 1, config.num_epochs, loss_avg, acc_1 / tt_1)

        CNNCRFModel.eval()
        valid_acc, valid_edge_acc, valid_std, valid_edge_std = valid.get_valid_acc(config, CNNCRFModel, valid_loader, epoch, test_mark='valid', save_acc=False)
        
        # 根据学习率调度器类型传入不同参数
        if config.lr_sche == 'RlrOP':
            lr_scheduler.step(valid_acc)  # ReduceLROnPlateau需要监控指标
        else:
            lr_scheduler.step()  # MultiStepLR不需要参数
        print_line += 'valid_acc: {:.5f} ± {:.5f}, '.format(valid_acc, valid_std)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch + 1


        test_acc_list = []
        for test_loader, test_mark in zip(test_loader_list, config.test_mark_list):
            test_acc, test_edge_acc, test_std, test_edge_std = valid.get_valid_acc(config, CNNCRFModel, test_loader, epoch, test_mark=test_mark, save_acc=config.save_acc)
            print_line += '{}_acc: {:.5f} ± {:.5f}, '.format(test_mark, test_acc, test_std)
            print_line += '{}_edge_acc: {:.5f} ± {:.5f}, '.format(test_mark, test_edge_acc, test_edge_std)
            test_acc_list.append(test_acc)

        end = datetime.datetime.now()
        print_line += 'be: {}, time: {}'.format(best_epoch, end-start)
        print(print_line)
        write_log.write(print_line)
        write_log.write('\n')
        write_log.flush()



        # 保存时始终保存真实模型参数，避免 DataParallel 前缀导致的兼容问题
        state = {'model': (CNNCRFModel.module.state_dict() if hasattr(CNNCRFModel, 'module') else CNNCRFModel.state_dict()),
                 'optimizer': optimizer.state_dict(),
                 'lr_scheduler': lr_scheduler.state_dict(),
                 'epoch': epoch}

        save_dpath = os.path.join(
            config.model_path, 'Epoch{}.ckpt'.format(epoch + 1))
        save_last_dpath = os.path.join(config.model_path, 'Epoch_last.ckpt')
        torch.save(state, save_dpath)
        torch.save(state, save_last_dpath)


