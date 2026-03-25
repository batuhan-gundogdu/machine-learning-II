import ssl
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class Data:
    def __init__(self, batch_size=16, test=False, size=800):
        self.batch_size = batch_size
        self.test = test
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = self.load_data()
        

    def load_data(self):
        # Disable SSL verification globally for this session to handle macOS SSL issues
        ssl._create_default_https_context = ssl._create_unverified_context
        
        train = not self.test
        dataset = FashionMNIST(root="./data", train=train, download=True, transform=None)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize((0.2860,), (0.3530,))])

        if not self.test:           
            indices = list(range(self.size))
            random.shuffle(indices)
            dataset = torch.utils.data.Subset(dataset, indices) 

        image_data = torch.stack([transform(img) for img, _ in dataset]).to(self.device)
        labels = torch.tensor([label for _, label in dataset]).to(self.device)
        dataset = TensorDataset(image_data, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  

        return dataloader

    def peek(self):
        class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        for batch_idx, (data, target) in enumerate(self.dataloader):
            print(data.shape)
            print(target.shape)
            # make a grid of 4x4 images
            fig, axs = plt.subplots(4, 4, figsize=(5, 5))
            for i in range(data.shape[0]):
                axs[i//4, i%4].imshow(data[i].squeeze().cpu().numpy(), cmap='gray')
                axs[i//4, i%4].set_title(class_names[target[i].item()])
                axs[i//4, i%4].axis('off')
            plt.show()
            break

    