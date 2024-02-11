import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Flatten():
    def __call__(self, x):
        return x.view(x.shape[0], -1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    # Flatten(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)


if __name__ == '__main__':
    for i, data in enumerate(trainloader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
        break

