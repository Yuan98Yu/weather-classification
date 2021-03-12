import torch
import torchvision.transforms as tt
from timm.data.random_erasing import RandomErasing
from timm.data import Mixup


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

tt2 = tt.Compose([
    tt.RandomResizedCrop([200, 200]),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(30),
    tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

tt3 = tt.Compose([
    tt.RandomResizedCrop([200, 200]),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(30),
    # tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

tt4 = tt.Compose([
    tt.Resize([256, 256]),
    tt.RandomCrop([200, 200]),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(30),
    # tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

tt5 = tt.Compose([
    tt.Resize([400, 400]),
    tt.RandomCrop([320, 320]),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(30),
    # tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

tt6 = tt.Compose([
    tt.Resize([400, 400]),
    tt.RandomCrop([256, 256]),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(30),
    # tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

tt7 = tt.Compose([
    tt.Resize([400, 400]),
    tt.RandomCrop([320, 320]),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(30),
    # tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    tt.ToTensor(),
    tt.Normalize(*stats),
    RandomErasing(device="cpu")
])

tt8 = tt.Compose([
    tt.Resize([400, 400]),
    tt.RandomCrop([320, 320]),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(30),
    # tt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    Mixup(num_classes=6),
    tt.ToTensor(),
    tt.Normalize(*stats)
])


vt1 = tt.Compose([tt.Resize([200, 200]), tt.ToTensor(), tt.Normalize(*stats)])
vt2 = tt.Compose([tt.Resize([320, 320]), tt.ToTensor(), tt.Normalize(*stats)])
vt3 = tt.Compose([tt.Resize([256, 256]), tt.ToTensor(), tt.Normalize(*stats)])
vt4 = tt.Compose([tt.Resize([400, 400]), tt.ToTensor(), tt.Normalize(*stats)])

classes0 = ['cloudy', 'haze', 'rainy', 'snow', 'sunny', 'thunder']
classes1 = ['cloudy', 'rainy', 'snow', 'sunny', 'thunder']
classes2 = ['haze', 'rainy', 'snow', 'sunny', 'thunder']
classes3 = ['cloudy', 'haze', 'snow', 'sunny', 'thunder']

efficientnet_b3a_e81_b16_tt7_vt2_explr_timmfc3clf_freeze_classes2_adabound = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt7',
    'valid_transform': 'vt2',
    'model': 'TimmFC3CLF',
    'freeze': True,
    'classes': 'classes2',
    'opt_func': 'adabound'
}

efficientnet_b3a_e81_b16_tt7_vt2_explr_WeatherModel3_freeze_classes1 = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt7',
    'valid_transform': 'vt2',
    'model': 'WeatherModel3',
    'freeze': True,
    'classes': 'classes1'
}

efficientnet_b3a_e81_b16_tt7_vt2_explr_WeatherModel3_freeze_classes2 = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt7',
    'valid_transform': 'vt2',
    'model': 'WeatherModel3',
    'freeze': True,
    'classes': 'classes2'
}

efficientnet_b3a_e81_b16_tt7_vt2_explr_WeatherModel3_freeze_classes3 = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt7',
    'valid_transform': 'vt2',
    'model': 'WeatherModel3',
    'freeze': True,
    'classes': 'classes3'
}

efficientnet_b3a_e81_b16_tt7_vt2_explr_WeatherModel3_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt7',
    'valid_transform': 'vt2',
    'model': 'WeatherModel3',
    'freeze': True
}

efficientnet_b3a_e81_b16_tt7_vt2_explr_WeatherModel5_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt7',
    'valid_transform': 'vt2',
    'model': 'WeatherModel5',
    'freeze': True
}

efficientnet_b3a_e81_b16_tt8_vt2_explr_WeatherModel3_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt8',
    'valid_transform': 'vt2',
    'model': 'WeatherModel3',
    'freeze': True
}

four_class_efficientnet_b3a_e81_b16_tt5_vt3_explr_WeatherModel3_freeze = {
    'exp_id': 'exp-4class_b3a',
    'data_dir': './data_tmp',
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt5',
    'valid_transform': 'vt3',
    'model': 'WeatherModel3',
    'freeze': True
}

efficientnet_b3a_e81_b16_tt5_vt3_explr_WeatherModel3_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 16,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt5',
    'valid_transform': 'vt3',
    'model': 'WeatherModel3',
    'freeze': True,
    'classes': 'classes0'
}

resnet34_e121_b128_tt5_vt2_explr_WeatherModel2_freeze = {
    'data_dir': data_dir,
    'epochs': 121,
    'batch_size': 128,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34',
    'pretrained': True,
    'train_transform': 'tt5',
    'valid_transform': 'vt2',
    'model': 'WeatherModel2',
    'freeze': True
}


resnet15_e151_b512_tt3_vt1_explr_resnet15_unfreeze = {
    'data_dir': data_dir,
    'epochs': 151,
    'batch_size': 512,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet15',
    'pretrained': False,
    'train_transform': 'tt3',
    'valid_transform': 'vt1',
    'model': 'WeatherModel4',
    'freeze': False
}


efficientb3a_e81_b32_tt3_vt1_explr_WeatherModel3_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 32,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'efficientnet_b3a',
    'pretrained': True,
    'train_transform': 'tt3',
    'valid_transform': 'vt1',
    'model': 'WeatherModel3',
    'freeze': True
}

regnetx16_e81_b128_tt3_vt1_explr_WeatherModel3_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 128,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'regnetx_016',
    'pretrained': True,
    'train_transform': 'tt3',
    'valid_transform': 'vt1',
    'model': 'WeatherModel3',
    'freeze': True
}

regnetx32_e81_b128_tt3_vt1_explr_WeatherModel3_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 128,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'regnetx_032',
    'train_transform': 'tt3',
    'valid_transform': 'vt1',
    'model': 'WeatherModel3',
    'freeze': True
}

resnet34_e81_b256_tt4_vt1_explr_WeatherModel1_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 256,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34',
    'train_transform': 'tt4',
    'valid_transform': 'vt1',
    'model': 'WeatherModel1',
    'freeze': True
}

resnet34_e81_b256_tt2_vt1_explr_WeatherModel1_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 256,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34',
    'train_transform': 'tt2',
    'valid_transform': 'vt1',
    'model': 'WeatherModel1',
    'freeze': True
}

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

resnet34_e81_b256_tt1_explr_WeatherModel1_freeze = {
    'data_dir': data_dir,
    'epochs': 81,
    'batch_size': 256,
    'max_lr': 3e-4,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'pretrained_model': 'resnet34',
    'train_transform': 'tt1',
    'model': 'WeatherModel1',
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
