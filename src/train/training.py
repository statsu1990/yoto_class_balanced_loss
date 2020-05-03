import torch
import numpy as np
from sklearn.metrics import f1_score

from .scheduler import WarmUpLR
from .train_utils import save_log, save_checkpoint
from tqdm import tqdm

dataset_is_cifar10 = True

class ScoreCalculator:
    def __init__(self):
        self.labels_true = []
        self.labels_pred = []

        self.n_major = 0
        self.n_correct_major = 0
        self.n_minor = 0
        self.n_correct_minor = 0

        # for cifar 10
        if dataset_is_cifar10:
            self.major_labels = np.array([0,2,4,6,8])
            self.minor_labels = np.array([1,3,5,7,9])
        # for cifar 100
        else:
            self.major_labels = np.arange(50)
            self.minor_labels = np.arange(50,100)

    def add_result(self, labels_true, labels_pred):
        self.labels_true.append(np.ravel(labels_true))
        self.labels_pred.append(np.ravel(labels_pred))

        cor_label = labels_true[labels_true==labels_pred]
        self.n_major += np.sum(np.isin(labels_true, self.major_labels))
        self.n_correct_major += np.sum(np.isin(cor_label, self.major_labels))
        self.n_minor += np.sum(np.isin(labels_true, self.minor_labels))
        self.n_correct_minor += np.sum(np.isin(cor_label, self.minor_labels))

    def calc_score(self):
        labels_true = np.concatenate(self.labels_true)
        labels_pred = np.concatenate(self.labels_pred)
        macro_f1 = f1_score(labels_true, labels_pred, average='macro')
        micro_f1 = f1_score(labels_true, labels_pred, average='micro')

        acc_major = self.n_correct_major / self.n_major
        acc_minor = self.n_correct_minor / self.n_minor
        acc_all = (self.n_correct_major + self.n_correct_minor) / (self.n_major + self.n_minor)
        return acc_all, acc_major, acc_minor, macro_f1, micro_f1

def trainer(net, loader, criterion, optimizer, grad_accum_steps, warmup_scheduler, use_yoto=False):
    net.train()

    total_loss = 0
    score_calculator = ScoreCalculator()

    optimizer.zero_grad()
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader)):
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        imgs = imgs.cuda()
        labels = labels.cuda()

        outputs = net(imgs)
        if use_yoto:
            #criterion_args = outputs[1].tolist()
            criterion_args = [outputs[1][:,i].unsqueeze(1) for i in range(outputs[1].size()[1])]
            outputs = outputs[0]
            loss = criterion(outputs, labels, *criterion_args)
        else:
            loss = criterion(outputs, labels)

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # loss
        total_loss += loss.item() * grad_accum_steps
        
        # score
        with torch.no_grad():
            score_calculator.add_result(labels.cpu().numpy(), outputs.max(1)[1].cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)

    # score
    acc_all, acc_major, acc_minor, macro_f1, micro_f1 = score_calculator.calc_score()

    print('Train Loss: %.3f | Acc: %.3f%%  AccMajor: %.3f%%  AccMinor: %.3f%%  MacroF1: %.3f%%  MicroF1: %.3f%% ' % (total_loss, acc_all, acc_major, acc_minor, macro_f1, micro_f1))
    
    return total_loss, acc_all, acc_major, acc_minor, macro_f1, micro_f1

def tester(net, loader, criterion, return_value=False, use_yoto=False):
    net.eval()
    total_loss = 0
    score_calculator = ScoreCalculator()

    true_label = []
    pred_label = []
    pred_logit = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader)):
            imgs = imgs.cuda()
            labels = labels.cuda()

            outputs = net(imgs)
            if use_yoto:
                #criterion_args = outputs[1].tolist()
                criterion_args = [outputs[1][:,i].unsqueeze(1) for i in range(outputs[1].size()[1])]
                outputs = outputs[0]
                if criterion is not None:
                    loss = criterion(outputs, labels, *criterion_args)
            else:
                if criterion is not None:
                    loss = criterion(outputs, labels)

            # loss
            if criterion is not None:
                total_loss += loss.item()
            else:
                total_loss += 0
        
            # score
            with torch.no_grad():
                score_calculator.add_result(labels.cpu().numpy(), outputs.max(1)[1].cpu().numpy())

            if return_value:
                true_label.append(labels.cpu().numpy())
                pred_label.append(predicted.cpu().numpy())
                pred_logit.append(outputs.cpu().numpy())

    # loss
    total_loss = total_loss / (batch_idx + 1)
    # score
    acc_all, acc_major, acc_minor, macro_f1, micro_f1 = score_calculator.calc_score()
    print('Test Loss: %.3f | Acc: %.3f%%  AccMajor: %.3f%%  AccMinor: %.3f%%  MacroF1: %.3f%%  MicroF1: %.3f%% ' % (total_loss, acc_all, acc_major, acc_minor, macro_f1, micro_f1))
    
    if return_value:
        true_label = np.concatenate(true_label)
        pred_label = np.concatenate(pred_label)
        pred_logit = np.concatenate(pred_logit)
        values = np.concatenate([true_label[:,None], pred_label[:,None], pred_logit], axis=1)
        return total_loss, acc_all, acc_major, acc_minor, macro_f1, micro_f1, values
    else:
        return total_loss, acc_all, acc_major, acc_minor, macro_f1, micro_f1

def train_model(net, tr_loader, vl_loader, 
                optimizer, tr_criterion, vl_criterion, 
                grad_accum_steps, start_epoch, epochs, 
                warmup_epoch, step_scheduler, filename_head='', use_yoto=False):
    net = net.cuda()

    # warmup_scheduler
    if start_epoch < warmup_epoch:
        warmup_scheduler = WarmUpLR(optimizer, len(tr_loader) * warmup_epoch)
    else:
        warmup_scheduler = None
    
    # train
    loglist = []
    for epoch in range(start_epoch, epochs):
        if epoch > warmup_epoch - 1:
            warm_sch = None
            step_scheduler.step()
        else:
            warm_sch = warmup_scheduler

        print('epoch ', epoch)
        for param_group in optimizer.param_groups:
            print('lr ', param_group['lr'])
            now_lr = param_group['lr']

        tr_log = trainer(net, tr_loader, tr_criterion, optimizer, grad_accum_steps, warm_sch, use_yoto)
        vl_log = tester(net, vl_loader, vl_criterion, return_value=False, use_yoto=use_yoto)

        # save checkpoint
        save_checkpoint(epoch, net, optimizer, filename_head + 'checkpoint')

        # save log
        loglist.append([epoch] + [now_lr] + list(tr_log) + list(vl_log))
        colmuns = ['epoch', 'lr', 'tr_loss', 'tr_acc_all', 'tr_acc_major', 'tr_acc_minor', 'tr_macro_f1', 'tr_micro_f1', 
                   'vl_loss', 'vl_acc_all', 'vl_acc_major', 'vl_acc_minor', 'vl_macro_f1', 'vl_micro_f1']
        save_log(loglist, colmuns, filename_head + 'training_log.csv')

    return net

def test_yoto(net, loader, 
              param_grids=((0.1, 0.5, 1.0), ),
              filename_head=''):

    net = net.cuda()

    prm_meshgrid = np.meshgrid(*param_grids)
    prm_meshgrid = [np.ravel(ms)[:,None] for ms in prm_meshgrid]
    prm_meshgrid = np.concatenate(prm_meshgrid, axis=1)

    scores = []
    for prm in prm_meshgrid:
        net.set_params(prm)
        total_loss, acc_all, acc_major, acc_minor, macro_f1, micro_f1 = tester(net, loader, None, return_value=False, use_yoto=True)
        print('prm: %s | Acc: %.3f%%  AccMajor: %.3f%%  AccMinor: %.3f%%  MacroF1: %.3f%%  MicroF1: %.3f%%' % (prm, acc_all, acc_major, acc_minor, macro_f1, micro_f1))
        scores.append([acc_all, acc_major, acc_minor, macro_f1, micro_f1])

    columns = ['param'+str(i) for i in range(prm_meshgrid.shape[1])]
    columns = columns + ['acc_all', 'acc_major', 'acc_minor', 'macro_f1', 'micro_f1']
    values = np.concatenate([prm_meshgrid, np.array(scores)], axis=1)
    save_log(values, columns, filename_head + 'yoto_result.csv')

    return
