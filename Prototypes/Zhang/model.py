import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, nc, version, batch_size, batch_normalize=False, initialize=True):
        super(VGG, self).__init__()
        self.version = version
        self.batch_size = batch_size
        self.batch_normalize = batch_normalize
        self.input_channels = 3
        self.vgg_structure = {
            9: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
            11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }

        self.net = self.__build_net()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),  # vgg16
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=nc))
        
        if initialize:
            self.init_bias(version)

    def __build_net(self):
        net = nn.Sequential()
        input_channels = self.input_channels
        for layer in self.vgg_structure[self.version]:
            if layer=='M':
                maxpool_layer = nn.MaxPool2d(2, 2)
                net.append(maxpool_layer)
            else:
                net.append(nn.Conv2d(input_channels, out_channels=layer, kernel_size=3, stride=1, padding=1))
                if self.batch_normalize:
                    net.append(nn.BatchNorm2d(num_features=layer))
                net.append(nn.ReLU())
                input_channels = layer
        return net

    def forward(self, inputs):
        result1 = self.net(inputs)
        result1 = result1.view(self.batch_size, -1)
        outputs = self.classifier(result1)

        return outputs

    
    def init_bias(self, version):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 1e-2)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                # nn.init.constant_(layer.weight, 0)
                nn.init.normal_(layer.weight, 0, 1e-2)
                nn.init.constant_(layer.bias, 0)