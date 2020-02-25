import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from siamese_net import SiameseNet
from celeb_loader import CelebLoader
import os
import numpy as np

torch.manual_seed(123)

def save_model(chpt):
    state = {'net': model.module.state_dict() if use_cuda else model.state_dict(),
             'epoch': epoch,
             'optimizer_state_dict': optimizer.state_dict()
             }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+chpt)

def train(model, loader, opt, epoch,criterion):
    model.train()
    correct = 0
    total_loss = 0
    for i, (speaker1, speaker2, label) in enumerate(loader):
        speaker1, speaker2, label = speaker1.cuda(), speaker2.cuda(), label.cuda()
        opt.zero_grad()
        output = model(speaker1.float(), speaker2.float())
        loss = criterion(output, label)
        total_loss += loss.item()
        loss.backward()
        opt.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        # acc = correct.item() / ((i + 1) * b) * 100.0
        # cur_acc = cur_correct.item() / BATCH_SIZE * 100
        # if i % 20 == 0:
        #     progress.set_description(f"Loss: {criterion}, CurAcc: {cur_acc}, Acc: {acc} ({correct}/{(i + 1) * BATCH_SIZE})")
    print(f"Epoch {epoch} Train Accuracy: {correct*100/len(loader.dataset)}({correct}/{len(loader.dataset)}), Loss: {loss/len(loader.dataset)}")
    return correct*100/len(loader.dataset)

def test(model, loader, epoch, criterion):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for i, (speaker1, speaker2, label) in enumerate(loader):
            speaker1, speaker2, label = speaker1.cuda(), speaker2.cuda(), label.cuda()
            output = model(speaker1.float(),speaker2.float())
            loss += criterion(output, label)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            # acc = correct.item() / ((i + 1) * batch_size) * 100.0
            # cur_acc = cur_correct.item() / BATCH_SIZE * 100
        print(f"Epoch {epoch} Test Accuracy:  {correct*100/len(loader.dataset)}({correct}/{len(loader.dataset)}), Loss: {loss/len(loader.dataset)} ({correct}/{len(loader.dataset)})")
    return correct*100/len(loader.dataset), loss/len(loader.dataset)

def load_data(batch_size):
    train_loader = torch.utils.data.DataLoader(
        CelebLoader("trainset", train=True), batch_size=batch_size, shuffle=True, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        CelebLoader("valset", train=False), batch_size=batch_size, shuffle=False, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(
        CelebLoader("testset", train=False), batch_size=1, shuffle=False, pin_memory=True, sampler=None)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    lr = 0.001
    batch_size = 32
    red = False
    best_acc = -1
    best_loss = 100
    print(f"lr {lr} batch {batch_size}")
    train_loader, val_loader, test_loader = load_data(batch_size)
    model = SiameseNet()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        # model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = F.cross_entropy#F.nll_loss
    for epoch in range(50):
        train(model,train_loader, optimizer, epoch, criterion)
#        save_model("nll_act")
        acc, loss = test(model, val_loader, epoch, criterion)
        if acc >= best_acc and loss <= best_loss:
            best_acc = acc
            best_loss = loss
            save_model("rnn_ce_2layer")
        #if acc >= 78 and red == False:
         #   red = True
          #  print("change lr")
           # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr*0.2
    test(model,test_loader, epoch, criterion)
