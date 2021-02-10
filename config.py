import torchvision.transforms as tt
from torchvision.transforms.transforms import ColorJitter, Compose, RandomApply, RandomRotation, RandomVerticalFlip


data_dir = './data'

stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

tt1 = tt.Compose([
    tt.RandomResizedCrop([200, 200]),
    tt.RandomHorizontalFlip(),
    tt.RandomVerticalFlip(),
    tt.RandomRotation(90),
    tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

res34_e81_b64_explr_tt1_model2_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 64,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34',
    'train_transform': 'tt1',
    'model': 'WeatherModel2',
    'freeze': True
}

res34_e181_b64_explr_tt1_model1_freeze = {
    'data_dir': data_dir,
    'epochs': 181,
    'batch_size': 64,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34',
    'train_transform': 'tt1',
    'model': 'WeatherModel1',
    'freeze': True
}

res34_e181_explr_tt1_model2 = {
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34',
    'train_transform': 'tt1',
    'model': 'WeatherModel2'
}

res34_e181_explr_tt1 = {
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34',
    'train_transform': 'tt1'
}

res50_e181_explr_tt1 = {
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet50',
    'train_transform': 'tt1'
}

res101_e181_explr_tt1 = {
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet101',
    'train_transform': 'tt1'
}

res34 = {
    'exp_id': 'exp-res34-1',
    'data_dir': data_dir,
    'epochs': 20,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34'
}

res34_2 = {
    'exp_id': 'exp-res34-2',
    'data_dir': data_dir,
    'epochs': 30,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34'
}

res34_3 = {
    'exp_id': 'exp-res34-3',
    'data_dir': data_dir,
    'epochs': 60,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34'
}

res34_4 = {
    'exp_id': 'exp-res34-4',
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 1e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34'
}

res34_e181_explr = {
    'exp_id': 'res34_e181_explr',
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34'
}

res34_e181_OClr = {
    'exp_id': 'res34_e181_OClr',
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34'
}

res50 = {
    'exp_id': 'exp-res50-1',
    'data_dir': data_dir,
    'epochs': 20,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet50'
}

res50_2 = {
    'exp_id': 'exp-res50-2',
    'data_dir': data_dir,
    'epochs': 30,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet50'
}

res50_3 = {
    'exp_id': 'exp-res50-3',
    'data_dir': data_dir,
    'epochs': 60,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet50'
}

res50_4 = {
    'exp_id': 'exp-res50-4',
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 1e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet50'
}

res50_e181_explr = {
    'exp_id': 'res50_e181_explr',
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet50'
}

res50_e181_OClr = {
    'exp_id': 'res50_e181_OClr',
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet50'
}

res101 = {
    'exp_id': 'exp-res101-1',
    'data_dir': data_dir,
    'epochs': 30,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet101'
}

res101_2 = {
    'exp_id': 'exp-res101-2',
    'data_dir': data_dir,
    'epochs': 60,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet101'
}

res101_3 = {
    'exp_id': 'exp-res101-3',
    'data_dir': data_dir,
    'epochs': 120,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet101'
}

res101_e181_explr = {
    'exp_id': 'exp_res101_e181_explr',
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet101'
}

res101_e181_OClr = {
    'exp_id': 'exp_res101_e181_OClr',
    'data_dir': data_dir,
    'epochs': 181,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet101'
}
