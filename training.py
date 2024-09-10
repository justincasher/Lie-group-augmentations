def training(deviation, complexPGL2Pct, realPGL2SquaredPct, realPGL3Pct, save_name) :
    """
    Trains a WideResNet model on the CIFAR-100 dataset with a
    given percentage of the data being augmented.

    :type deviation: float
        How large of transformations to generate
    :type complexPGL2Pct: int
        What percent of the images to apply a PGL(C, 2) transformation to
    :type realPGL2SquaredPct: int
        What percent of the images to apply a PGL(R, 2)^2 transformation to
    :type realPGL3Pct: int
        What percent of the images to apply a PGL(R, 3) transformation to
    """

    # includes: general
    import numpy as np
    import random
    from time import time
    
    # includes: torch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms 

    # import models
    from models.wide_resnet import WideResNet

    # includes: data augmentation
    from augmentor import augment

    # data loading
    batch_size = 128
    image_size = 32

    # import training data
    # we do not normalize until after augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True,
        download=False, 
        transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=True
    ) 

    # import testing data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR100(
        root='./data', 
        train=False,
        download=False, 
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size,
        shuffle=False
    ) 

    # network architecture
    net = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
    net.to(device)
    
    # optimizer choices
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, weight_decay=0.0005, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

    # training procedure
    start = time()

    write_path = "outputs/" + save_name + ".txt"
    f = open(write_path, "a")
    f.write("Epoch, accuracy, average time\n")
    f.close()

    print("Epoch, accuracy, average time")

    for epoch in range(200) :        
        # set net to train to enable batch norm
        net.train()
            
        # training procedure
        for _, data in enumerate(train_loader, 0):
            # load data
            images, labels = data

            # augment data
            augment(images, image_size, deviation, complexPGL2Pct, realPGL2SquaredPct, realPGL3Pct)
            
            # pass to device
            images = images.to(device)
            labels = labels.to(device)

            # pass augmented data through net
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # adjust the learning rate
        scheduler.step()
            
        # testing procedure 
        if (epoch+1) % 5 == 0 and epoch > 0 : 
            net.eval()

            # compute time per epoch
            time_diff = round(time() - start, 2)
            average_time = round(time_diff / (epoch+1), 2)

            # compute accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    # pass to device 
                    images = images.to(device)
                    labels = labels.to(device)

                    # see if outputs are accurate
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
            accuracy = round(100 * correct / total, 2)
            
            write_path = "outputs/" + save_name + ".txt"
            f = open(write_path, "a")
            f.write(f"{epoch+1} \t {accuracy} \t {average_time}\n")
            f.close()

    # save network
    path = "networks/" + save_name
    torch.save(net, path)
