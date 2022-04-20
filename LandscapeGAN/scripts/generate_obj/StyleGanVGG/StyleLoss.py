import torch
import torchvision
import torch.nn as nn



class VGGStyleLoss(torch.nn.Module):
    def __init__(self, transfer_mode, resize=True):
        super(VGGStyleLoss, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        blocks = []
        if transfer_mode == 0:  # transfer color only
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
        else: # transfer both color and texture
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
            blocks.append(vgg.features[9:16].eval())
            blocks.append(vgg.features[16:23].eval())

        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = resize

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * w * h)
        return gram

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            gm_x = self.gram_matrix(x)
            gm_y = self.gram_matrix(y)
            loss += torch.sum((gm_x-gm_y)**2)
        return loss
    