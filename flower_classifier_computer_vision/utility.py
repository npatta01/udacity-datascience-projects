from torchvision import datasets, transforms, models
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45)
            ,transforms.RandomResizedCrop(224)
            ,transforms.RandomHorizontalFlip()
            ,transforms.ToTensor()
            ,transforms.Normalize(imagenet_mean,imagenet_std)
        ])
        ,'valid' : transforms.Compose([
            transforms.Resize(255)
            ,transforms.CenterCrop(224)
            ,transforms.ToTensor()
            ,transforms.Normalize(imagenet_mean,imagenet_std)
        ])
        ,'test' : transforms.Compose([
            transforms.Resize(255)
            ,transforms.CenterCrop(224)
            ,transforms.ToTensor()
            ,transforms.Normalize(imagenet_mean,imagenet_std)
        ])
    }

