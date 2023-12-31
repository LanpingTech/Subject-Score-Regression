import torch

class CrossNetwork(torch.nn.Module):
    def __init__(self, input_dim, layer_num=3):
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList([torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, 1))) for _ in range(layer_num)])

    def forward(self, x):
        x_0 = x
        for i in range(self.layer_num):
            x_w = torch.tensordot(x, self.kernels[i], dims=([1], [0]))
            x = x_0 * x_w + x
        return x
    
class DeepNetwork(torch.nn.Module):
    def __init__(self, input_dim, layer_num=3):
        super(DeepNetwork, self).__init__()

        self.layer_num = layer_num
        self.linears = torch.nn.Sequential()
        for i in range(layer_num):
            self.linears.add_module('linear_{}'.format(i), torch.nn.Linear(input_dim, input_dim))
            self.linears.add_module('relu_{}'.format(i), torch.nn.ReLU())


    def forward(self, x):
        return self.linears(x)
    
class DCN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=3):
        super(DCN, self).__init__()

        self.to_embed = torch.nn.Embedding(3, 10)
        input_dim += 9

        self.cross_network = CrossNetwork(input_dim, layer_num)
        self.deep_network = DeepNetwork(input_dim, layer_num)

        self.regressor = torch.nn.Linear(input_dim * 2, 1)
        self.classifier = torch.nn.Linear(input_dim * 2, 4)

    def forward(self, x):
        x_discrete = self.to_embed(x[:, :1].long()).squeeze(1)
        x = torch.cat([x_discrete, x[:, 1:]], dim=1)
        x_cross = self.cross_network(x)
        x_deep = self.deep_network(x)
        x = torch.cat([x_cross, x_deep], dim=1)
        return self.regressor(x), self.classifier(x)


class DCN1(torch.nn.Module):
    def __init__(self, in_dim_cate, in_dim_num, out_dim_cate, out_dim_num, layer_num=3):
        super(DCN1, self).__init__()
        self.in_dim_cate = in_dim_cate
        self.in_dim_num = in_dim_num
        self.out_dim_cate = out_dim_cate
        self.out_dim_num = out_dim_num
        # 每个输入离散值转化成10维向量
        self.cate_dim = 10

        self.cross_network = CrossNetwork(len(in_dim_cate) * self.cate_dim + in_dim_num, layer_num)
        self.deep_network = DeepNetwork(len(in_dim_cate) * self.cate_dim + in_dim_num, layer_num)

        self.to_embeds = []
        for i in range(len(in_dim_cate)):
            self.to_embeds.append(torch.nn.Embedding(in_dim_cate[i], self.cate_dim))
        self.classifiers = []
        for i in range(len(out_dim_cate)):
            self.classifiers.append(torch.nn.Linear((len(in_dim_cate) * self.cate_dim + in_dim_num) * 2, out_dim_cate[i]))
        self.regressors = []
        for i in range(out_dim_num):
            self.regressors.append(torch.nn.Linear((len(in_dim_cate) * self.cate_dim + in_dim_num) * 2, 1))

    def forward(self, x):
        x_discrete = []
        for i in range(len(self.in_dim_cate)):
            x_discrete.append(self.to_embeds[i](x[:, i:i+1].long()).squeeze(1))

        x_discrete.append(x[:, len(self.in_dim_cate):])
        x = torch.cat(x_discrete, dim=1)

        x_cross = self.cross_network(x)
        x_deep = self.deep_network(x)
        x = torch.cat([x_cross, x_deep], dim=1)

        arr = []
        for i in range(len(self.out_dim_cate)):
            arr.append(self.classifiers[i](x))
        for i in range(self.out_dim_num):
            arr.append(self.regressors[i](x))
        return tuple(arr)
