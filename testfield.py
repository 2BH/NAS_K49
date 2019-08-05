from utils.datasets import K49
import torchvision.transforms as transforms

data_dir = './data'
data_augmentations = None
if data_augmentations is None:
    data_augmentations = transforms.ToTensor()
elif isinstance(type(data_augmentations), list):
    data_augmentations = transforms.Compose(data_augmentations)
elif not isinstance(data_augmentations, transforms.Compose):
    raise NotImplementedError
train_data = K49(data_dir, True, data_augmentations)
test_data = K49(data_dir, False, data_augmentations)

print(train_data.class_frequency)
print(test_data.class_frequency)