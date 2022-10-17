import sys
import logging

import torch
import numpy as np
import torchvision.datasets as datasets
import torch.utils.data as data

from .search import *
from .recovery import *
from .utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ["Trainer"]


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_ops      = len(PRIMITIVES)
        self.proj_dims    = cfg.proj_dims
        self.sparseness   = cfg.sparseness
        self.steps        = cfg.steps

        self.search_trainer = InnerTrainer(cfg)
        self.num_edges = self.search_trainer.model.num_edges
        self.train_queue, self.valid_queue = self.set_dataloader()#训练和验证的数据集的划分

    def set_dataloader(self):
        train_transform, valid_transform = cifar10_transforms(self.cfg)
        train_data = datasets.CIFAR10(
            root=self.cfg.data, train=True, download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.cfg.train_portion * num_train))

        train_queue = data.DataLoader(
            train_data, batch_size=self.cfg.batch_size,
            sampler=data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)

        valid_queue = data.DataLoader(
            train_data, batch_size=self.cfg.batch_size,
            sampler=data.sampler.SubsetRandomSampler(indices[split:]),
            pin_memory=True, num_workers=2)

        return train_queue, valid_queue
    #AS为矩阵A，alpha表示b
    def do_recovery(self, As, alpha):
        xs = []
        for i in range(self.steps):
            lasso = LASSO(As[i].cpu().numpy().copy())#lasso.A 的具体值与AS之中的对应tensor是一样的
            b = alpha[i]#属于B，z稀疏编码后的值
            x = lasso.solve(b)
            xs.append(x)
            #xs就是一轮训练和验证后，结构系数进行更新的结果

        return xs

    def do_search(self, A_normal, normal_biases,
                       A_reduce, reduce_biases, epoch):
        self.search_trainer.model.init_proj_mat(A_normal, A_reduce)
        self.search_trainer.model.init_bias(normal_biases, reduce_biases)#这个地方具体初始化了哪里
        # train
        train_acc, train_obj = self.search_trainer.train_epoch(
            self.train_queue, self.valid_queue, epoch)
        logging.info("train_acc {:.4f}".format(train_acc))
        # valid
        valid_acc, valid_obj = self.search_trainer.validate(self.valid_queue)
        logging.info("valid_acc {:.4f}".format(valid_acc))
        #第一个是self.alphas_normal_，第二个是self.alphas_reduce_,上一轮对应的权重
        alpha_normal, alpha_reduce = self.search_trainer.model.arch_parameters()#vaild没有改变权重值
        alpha_normal = alpha_normal.detach().cpu().numpy()#对self.alphas_normal_做近似处理
        alpha_reduce = alpha_reduce.detach().cpu().numpy()
        #detach()返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算，
        return alpha_normal, alpha_reduce
#等式8
    def sample_and_proj(self, base_As, xs):
        As= []
        biases = []
        for i in range(self.steps):
            A = base_As[i].numpy().copy()
            E = A.T.dot(A) - np.eye(A.shape[1])
            x = xs[i].copy()
            #这个地方是如何使用这个稀疏系数的
            zero_idx = np.abs(x).argsort()[:-self.sparseness]
            x[zero_idx] = 0.#另z中的对应位置变为0
            A[:, zero_idx] = 0.#同样对于A也进行稀疏化
            As.append(torch.from_numpy(A).float())
            E[:, zero_idx] = 0.#公式八
            bias = E.T.dot(x).reshape(-1, self.num_ops)#.dot两元素的乘积
            biases.append(torch.from_numpy(bias).float())

        biases = torch.cat(biases)
        # 这个函数返回的是Xi
        #AS返回的是b=A*Z的矩阵，就是用于稀疏化的矩阵
        return As, biases

    def show_selected(self, epoch, x_normals, x_reduces):
        print("[Epoch {}]".format(epoch if epoch > 0 else 'initial'))

        print("normal cell:")
        gene_normal = []#列出数据和对应下标
        #np.abs(x).argsort()
        #([ 4,  3,  6, 11, 10,  8, 13,  9,  5,  7,  0,  2, 12,  1])
        #x中从小到大排序中，最后两个的下标，-2，-1
        for i, x in enumerate(x_normals):
            id1, id2 = np.abs(x).argsort()[-2:]#14，21，28，35.7*14
            print("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            gene_normal.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))#num_ops怎么算的
            gene_normal.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        print("reduction cell:")
        gene_reduce = []
        for i, x in enumerate(x_reduces):
            id1, id2 = np.abs(x).argsort()[-2:]
            print("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            gene_reduce.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_reduce.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        concat = range(2, 2 + len(x_normals))
        genotype = Genotype(
            normal = gene_normal, normal_concat = concat,
            reduce = gene_reduce, reduce_concat = concat)
        print(genotype)
        model_cifar = NetworkCIFAR(16, 10, 8, False, genotype)#与two-stage不一样
        torch.save(model_cifar.cells._modules['0'],'/root/data/CELL.pt')
        param_size = count_parameters_in_MB(model_cifar)
        logging.info('param size = %fMB', param_size)

    def train(self):
        #A是完成稀疏编码所需要的矩阵等式8
        base_A_normals = []
        base_A_reduces = []

        for i in range(self.steps):
            base_A_normals.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))# 7*14，7*21，7*28，7*35
            base_A_reduces.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))
#torch.datech返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
# 不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度
        alpha_normal = self.search_trainer.model.alphas_normal_.detach().cpu().numpy()#4*(7,1)
        alpha_reduce = self.search_trainer.model.alphas_reduce_.detach().cpu().numpy()#原始空间后稀疏编码后的空间
        x_normals = self.do_recovery(base_A_normals, alpha_normal)#4*(14*1,21*1,28*1,35*1)
        x_reduces = self.do_recovery(base_A_reduces, alpha_reduce)#超级网络中连接的更新

        self.show_selected(0, x_normals, x_reduces)

        for i in range(self.cfg.epochs):
            #这里A_normals指的是将base_A_normals稀疏化编码后的结果
            #normal_biases指的是将更新后的z进行稀疏化后的结果
            A_normals, normal_biases = self.sample_and_proj(
                base_A_normals, x_normals)#将其原始空间转换为稀疏编码问题
            A_reduces, reduce_biases = self.sample_and_proj(
                base_A_reduces, x_reduces)
            print("Doing Search ...")
            alpha_normal, alpha_reduce = self.do_search(A_normals, normal_biases,
                                                       A_reduces, reduce_biases, i+1)
            print("Doing Recovery ...")#alpha_normal是4*(7*1)
            x_normals = self.do_recovery(base_A_normals, alpha_normal)#更新后的原始空间
            x_reduces = self.do_recovery(base_A_reduces, alpha_reduce)
            self.show_selected(i+1, x_normals, x_reduces)
        torch.save(self.search_trainer.model.alphas_normal,'/root/data/alphas.pt')
        
        #torch.save(self.search_trainer.outer_trainer.model, "my_model.pth")
        
        x = torch.rand(256, 3, 32, 32)
        model = torch.jit.trace(self.search_trainer.model.to(device), x.cuda())
        torch.onnx.export(model.to(device), # 搭建的网络
                        x.cuda(), # 输入张量
                        'targetmodel.onnx', # 输出模型名称
                        input_names=["input"], # 输入命名
                        output_names=["output"], # 输出命名
                       dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
        )
