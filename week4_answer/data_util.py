import torch
import torch.utils.data as data
import torchvision.datasets as vision_dataset
import torchvision.transforms as transforms

from .hparams import HyperParameters

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init()
        self.ROOT_DIR = './data'

        self.mnist_dataset = vision_dataset.MNIST(self.ROOT_DIR, download=True)

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, labe = self.mnist_data[idx]
        image = transforms.ToTensor()(image).squeeze(0) #[1, 28, 28]에서 1을 필요 없으므로 squeeze로 1을 뺀다.
        label = torch.LonTensor([label]).squeeze(0) #나중에 묶고 싶어서 1차원을 0차원으로 squeezing
        return image, label

def get_dataloader(hparam):
    return data.DataLoader(Dataset(), batch_size=hparam.batch_size, shuffle=hparam.shuffle, num_workers=hparam.num_workers)


if __name__ == "__main__":
    hparam = HyperParameters()
    dataloader = iter(get_dataloader(hparam=hparam))

    image, label = next(dataloader)