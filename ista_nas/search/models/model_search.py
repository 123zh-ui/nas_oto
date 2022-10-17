import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..operations import *
from ..genotypes import Genotype, PRIMITIVES
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

__all__ = ["MixedOp", "Cell", "NetWork"]


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        k=1
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            torch.save(op,'/root/data/Mop'+str(k)+'.pt')
            k=k+1
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                torch.save(op,'/root/data/cop.pt')
            self._ops.append(op)
     #针对一个mixop进行操作，也就是经过op处理然后乘上w   
    #x是数据，w是权重，w是tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0014, 0.0000],稀疏矩阵，大小是1*7
    def forward(self, x, weights):
        if weights.sum() == 0:
            return 0
        Temp=0
        for w,op in zip(weights.to(device), (self._ops).to(device)):
            op=op.to(device)
            if w !=0:
                Temp=w*op(x)+Temp
        return Temp
        #return sum(w * op(x) for w, op in zip(weights.to(device), (self._ops).to(device))if w != 0)#zip转换为元组
#0 Seq,1 seq,2 iden,3 seqconv,4 seqconv,5,dilconv,6 dilconv-->ops
#这个return后面就是使一个mixop中的每一个操作也就是上一行的每一个具体操作乘以对应位置的权值，也就是每次至多只能够有一个操作有效
class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
#OrderedDict([('0', ReLU()), 
# ('1', Conv2d(48, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)), 
# ('2', BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True))])
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)#当前cell的前前一个
            torch.save(self.preprocess0,'/root/data/preprocess0.pt')##与normal的相比，Factor中间是两个卷积，且输出是正常情况下的一半
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)#C_prev_prev表示卷积输入的大小，C表示卷积输出的大小
            torch.save(self.preprocess0,'/root/data/preprocess0.pt')
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)#BN能够加速网络收敛
        self._steps = steps
        self._multiplier = multiplier
        #这里也normal和reduce就有一个不同，也就是具体操作的时候步长的不同
        #也就是每一层前两个的步长参数不同
        self._ops = nn.ModuleList()
        k=1
        for i in range(self._steps):
            #i=0，j=0,1,;i=1,j=0,1,2;i=2,j=0,1,2,3;i=3,j=0,1,2,3,4
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                torch.save(op,'/root/data/op'+str(k)+'.pt')
                k = k+1#k最终为14
                self._ops.append(op)
        torch.save(self._ops,'/root/data/ops.pt')    
    #weights，torch.Size([14, 7])，weights每一行一般就只有一个有效值，其它都为0，同时也有整行都为0
    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0).to(device)#刚好是一个cell的内部结构,当是reduce时，其大小torch.Size([256, 32, 16, 16])
        s1 = self.preprocess1(s1).to(device)#torch.Size([256, 32, 32, 32])，当reduce时，torch.Size([256, 128, 16, 16])
        #14*7，21*7，28*7，35*7--》2，3，4，5
        states = [s0, s1]#这里states是一个长度为2 的队列
        offset = 0#这里的ops指的是一个cell的所有操作，包括14个mixop，每个mixop包括7个具体的操作
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h.to(device), weights[offset+j].to(device)) for j, h in enumerate(states))
            offset += len(states)#h的大小torch.Size([256, 32, 32, 32])
            states.append(s)#其中最开始的两个就是S0，S1，前一个时reduce后，大小就变为torch.Size([256, 32, 16, 16])
        #torch.Size([256, 16, 32, 32])*6--》states一个cell后的大小，
        return torch.cat(states[-self._multiplier:], dim=1)#取下了后四个，也就是op处理后的结果


class NetWork(nn.Module):

    def __init__(self, C, num_classes, layers,
                 proj_dims=2, steps=4, multiplier=4, stem_multiplier=3):
        super(NetWork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.num_edges = sum(1 for i in range(self._steps) for n in range(2 + i))#边的数目为14
        self.proj_dims = proj_dims

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))
        # 当前结点，相邻前一个结点，相邻的前两个结点值的更新
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        #8个cell，6个normal和2个reduce
        for i in range(layers):
            #[layers//3, 2*layers//3]---》[2, 5]也就是第三个和第六个是
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2#一般为normal的情况其实不进行更新的
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)#结构单元的产生
            #判断当前cell的前一个是否为reduce
            reduction_prev = reduction  # C_prev_prev， C_curr是0中卷积的输入和输出        
            self.cells.append(cell)# C_prev, C_curr是1中卷积的输入和输出
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            torch.save(cell,'/root/data/cell'+str(i)+'.pt')
        torch.save(self.cells,'/root/data/cells.pt')
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = NetWork(self._C, self._num_classes, self._layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    #一次迭代，input的大小torch.Size([256, 3, 32, 32])
    def forward(self, input):
        s0 = s1 = self.stem(input).to(device)#torch.Size([256, 48, 32, 32])，s0和s1是一样的
        #A = torch.tensor( [item.detach().numpy() for item in self.A_normals] )
        #B = torch.tensor( [item.detach().numpy() for item in self.A_reduces] )
        self.proj_alphas(self.A_normals, self.A_reduces)#一次迭代的时候weights始终没有变化
        for i, cell in enumerate(self.cells):
            cell=cell.to(device)
            if cell.reduction:
                weights = self.alphas_reduce.to(device)#权重，i=0....7
            else:
                weights = self.alphas_normal.to(device)#reduce后S0的大小torch.Size([256, 128, 16, 16])，S1为torch.Size([256, 128, 16, 16])
            s0, s1 = s1, cell(s0, s1, weights).to(device)#S1的torch.Size([256, 64, 32, 32])
        out = self.global_pooling(s1).to(device)#S0的torch.Size([256, 64, 32, 32])
        logits = self.classifier(out.view(out.size(0),-1)).to(device)#out的维度torch.Size([256, 256, 1, 1])
        #第二次后reduc后S0的大小torch.Size([256, 256, 8, 8])，S1为torch.Size([256, 256, 8, 8])
        return logits.to(device)#logits维度torch.Size([256, 10])

    def _loss(self, input, target):
        logits = self(input).to(device)#target的大小torch.Size([256])
        return F.cross_entropy(logits, target)

    def _initialize_alphas(self):
        self.alphas_normal_ = nn.Parameter(1e-3*torch.randn(self._steps, self.proj_dims))
        self.alphas_reduce_ = nn.Parameter(1e-3*torch.randn(self._steps, self.proj_dims))
        self._arch_parameters = [
            self.alphas_normal_,
            self.alphas_reduce_,
        ]#4*7

    def init_proj_mat(self, A_normals, A_reduces):
        self.A_normals = A_normals
        self.A_reduces = A_reduces

    def init_bias(self, normal_bias, reduce_bias):
        self.normal_bias = normal_bias
        self.reduce_bias = reduce_bias

#    def init_alphas(self, alphas_normals, alphas_reduces):
#        state_dict = self.state_dict()
#        new_state_dict = {}
#        for k, v in state_dict.items():
#            if 'alpha' not in k:
#                new_state_dict[k] = v
#            else:
#                if 'normal' in k:
#                    new_state_dict[k] = alphas_normals.to(v.device)
#                else:
#                    new_state_dict[k] = alphas_reduces.to(v.device)
#        self.load_state_dict(new_state_dict)
    #这个地方是A对应的normal和reduce稀疏后的A，即大部分都变为0
    def proj_alphas(self, A_normals, A_reduces):
        assert len(A_normals) == len(A_reduces) == self._steps
        alphas_normal = []
        alphas_reduce = []
        #tensor([[ 0.0002, -0.0022,  0.0008,  0.0003,  0.0010,  0.0006,  0.0002],
       # [ 0.0002, -0.0004,  0.0004,  0.0003, -0.0008, -0.0010, -0.0016],
        #[-0.0009, -0.0006, -0.0018,  0.0004, -0.0004, -0.0007,  0.0008],
        #[-0.0007, -0.0007,  0.0005, -0.0007,  0.0004, -0.0004,  0.0006]],
         #device='cuda:0')
        alphas_normal_ = self.alphas_normal_.to(device) #F.softmax(self.alphas_normal_, dim=-1)
        alphas_reduce_ = self.alphas_reduce_.to(device) #F.softmax(self.alphas_reduce_, dim=-1)
        for i in range(self._steps):#指的是一个cell中的后四个op进行的操作
            A_normal = A_normals[i].to(alphas_normal_.device).requires_grad_(False)
            #t_alpha的值tensor([[ 0.0000, -0.0010,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            # [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0020,  0.0000]],
             #device='cuda:0', grad_fn=<ReshapeAliasBackward0>)
             #.mm矩阵相乘
            t_alpha = alphas_normal_[[i]].mm(A_normal).reshape(-1, len(PRIMITIVES))
            alphas_normal.append(t_alpha.to(device))
            A_reduce = A_reduces[i].to(alphas_reduce_.device).requires_grad_(False)
            t_alpha = alphas_reduce_[[i]].mm(A_reduce).reshape(-1, len(PRIMITIVES))
            alphas_reduce.append(t_alpha.to(device))
        #上面那个循环是对normal和reduce都进行A*b的操作，马上回去的时候再看一下alphas_指的是什么
        self.alphas_normal = torch.cat(alphas_normal).to(device) - self.normal_bias.to(alphas_normal_.device)
        self.alphas_reduce = torch.cat(alphas_reduce).to(device) - self.reduce_bias.to(alphas_reduce_.device)#.device返回位置
        #前后两次的alphas_reduce的差值，normal和reduce都有，回去查一下_bias_

    def arch_parameters(self):
        return self._arch_parameters#第一个是self.alphas_normal_，第二个是self.alphas_reduce_

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                        if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
