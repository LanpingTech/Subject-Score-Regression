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

