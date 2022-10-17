import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from .outer_trainer import OuterTrainer
from .models import NetWork
from ..utils import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

__all__ = ["InnerTrainer"]

# 整个网络的一个初始化，包括网络结点，优化器，scheduler还有数据的划分
class InnerTrainer:
    def __init__(self, cfg):
        self.grad_clip = cfg.grad_clip
        self.report_freq = cfg.report_freq
        self.model = NetWork(cfg.init_channels, cfg.num_classes, cfg.layers, proj_dims=cfg.proj_dims).cuda()
        print("Param size = {}MB".format(count_parameters_in_MB(self.model)))
        torch.save(self.model,'/root/data/model.pt')
        weights = []#权重的初始化
        i=1
        t=[]
        for k, p in self.model.named_parameters():
            i = i+1#1401,出结构系数外
            t.append(k)
            if 'alpha' not in k:
                weights.append(p)#len(weights)=1399
        self.optimizer = optim.SGD(
            weights, cfg.learning_rate,
            momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )#SGD随机梯度下降，优化器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(cfg.epochs), eta_min=cfg.learning_rate_min
        )#scheduler是 对优化器的学习率进行调整 
        self.outer_trainer = OuterTrainer(self.model, cfg)

    def train_epoch(self, train_queue, valid_queue, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #scheduler是 对优化器的学习率进行调整 
        self.scheduler.step()
        lr = self.scheduler.get_lr()
        #arlr = self.outer_trainer.scheduler.get_lr()
        print('epoch: ', epoch, 'lr:', lr)
        valid_loader = iter(valid_queue)

        self.model.train()#train_queue中的shape:(50000, 32, 32, 3)
        for batch_id, (input, target) in enumerate(train_queue):
            # for inner update
            input = input.cuda()#torch.Size([256, 3, 32, 32])
            target = target.cuda()#256
            # for outer update
            try:
                input_search, target_search = next(valid_loader)
            except StopIteration:
                valid_loader = iter(valid_queue)
                input_search, target_search = next(valid_loader)

            input_search = input_search.cuda()#torch.Size([256, 3, 32, 32])，
            target_search = target_search.cuda()#256
            #整个network进行训练
            self.outer_trainer.step(input_search, target_search)#这个地方进行权重的更新

            self.optimizer.zero_grad()#清空过往的梯度
            
            scores = self.model(input).to(device)#两个的那四个都是一样的，A_normmal和A_reduce也是一样的loss = F.cross_entropy(scores, target)#与out一样
            loss = F.cross_entropy(scores, target).to(device)
            loss.backward()#更新外层网络参数的梯度
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)#grad_clip为5
            self.optimizer.step()#SGD。外层的训练

            n = input.size(0)#256
            prec1, prec5 = accuracy(scores, target, topk=(1, 5))#准确率
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if batch_id % self.report_freq == 0:
                print("Train[{:0>3d}] Loss: {:.4f} Top1: {:.4f} Top5: {:.4f}".format(
                    batch_id, losses.avg, top1.avg, top5.avg
                ))

            # Export the model
            #model = torch.jit.trace(self.model.to(device), input)#如果模型中存在循环或者if语句，
 #           torch.onnx.export(self.model.to(device),               # 在执行torch.onnx.export之前先使用torch.jit.script将nn.Module转换为ScriptModule
 #                         input.to(device),                         # model input (or a tuple for multiple inputs)
 #                          "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
 #                          export_params=True,        # store the trained parameter weights inside the model file
 #                          opset_version=10,          # the ONNX version to export the model to
 #                         do_constant_folding=True,  # whether to execute constant folding for optimization
 #                           input_names = ['input'],   # the model's input names
 #                          output_names = ['output'], # the model's output names
 #                           dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
 #                                           'output' : {0 : 'batch_size'}})
#        self.scheduler.step()
#        self.outer_trainer.scheduler.step()
        return top1.avg, losses.avg

    def validate(self, valid_queue):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for batch_id, (input, target) in enumerate(valid_queue):
                input = input.cuda()#一轮
                target = target.cuda()

                scores = self.model(input)
                #torch.onnx.export(self.model.to(device),               # 在执行torch.onnx.export之前先使用torch.jit.script将nn.Module转换为ScriptModule
                #input.to(device),                         # model input (or a tuple for multiple inputs)
                #"1234.onnx",   # where to save the model (can be a file or file-like object)
                #export_params=True,        # store the trained parameter weights inside the model file
                #opset_version=10,          # the ONNX version to export the model to
                #do_constant_folding=True,  # whether to execute constant folding for optimization
                #input_names = ['input'],   # the model's input names
                #output_names = ['output'], # the model's output names
                #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                 #               'output' : {0 : 'batch_size'}})
                loss = F.cross_entropy(scores, target)

                n = input.size(0)
                prec1, prec5 = accuracy(scores, target, topk=(1, 5))
                losses.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if batch_id % self.report_freq == 0:
                    print(" Valid[{:0>3d}] Loss: {:.4f} Top1: {:.4f} Top5: {:.4f}".format(
                        batch_id, losses.avg, top1.avg, top5.avg
                    ))

        return top1.avg, losses.avg
