import torchvision.transforms as transforms


def simple_transform(cfg, is_train):
    mean = cfg.mean
    std = cfg.std
    # if is_train is True:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    # else:
    #     transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=mean, std=std),
    #         ]
    #     )
    return transform
