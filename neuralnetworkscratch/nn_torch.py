from torch import nn

class MyNeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyNeuralNet,self).__init__()
        self.nn = nn.Linear(input_size, output_size)
        print(self.nn.weight)
        print(self.nn.bias)

    def forward(self, inputs):
        pass


model = MyNeuralNet(16,2)
