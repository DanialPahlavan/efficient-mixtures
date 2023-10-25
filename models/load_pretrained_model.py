import torch
from models.resnets import resnet20, resnet1202


def load_resnet(model_name):
    path = '/home/oskar/phd/efficient_mixtures/models/pretrained_models'
    if model_name == 'resnet20':
        path += '/resnet20-12fca82f.th'
        resnet = resnet20()
    elif model_name == 'resnet1202':
        path += '/resnet1202-f3b1deed.th'
        resnet = resnet1202()
    old_state_dict = torch.load(path)['state_dict']
    new_state_dict = {}
    for name in old_state_dict.keys():
        # remove module. to align with expected state_dict by model
        new_name = name[7:]
        new_state_dict[new_name] = old_state_dict[name]
    resnet.load_state_dict(new_state_dict)
    return resnet


if __name__ == '__main__':
    resnet = load_resnet('resnet20')
    x = resnet(torch.zeros(1, 3, 32, 32))
    print(x.view(64).shape)
