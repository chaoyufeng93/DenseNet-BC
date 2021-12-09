Data Augmentation: RandomCrop(32, padding=4) & transforms.RandomHorizontalFlip()

SGD: weight_decay = 1e-4,momentum = 0.9,nesterov=True

Learning rate: 0.1, 0.01(100th epoch), 0.001(150th epoch)

model parameter setting: mod = DenseNet_Pro(12,[16,16,16],0.5,0.2) # DenseNet-BC 100

acc: 93.83%
