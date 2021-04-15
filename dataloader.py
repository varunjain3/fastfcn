
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from configs import Configs


class CustomDataset(Dataset):

    def __init__(self, configs: Configs):

        self.device = configs.device
        self.path = configs.datasetPath
        self.size = (configs.image_size, configs.image_size)

        self.data = torch.load(self.path)
        self.length = len(self.data)

        print(self.length, 'images in', self.path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        if index > self.length:
            raise Exception(
                f"Dataloader out of index. Max Index:{self.length - self.n_images}, Index asked:{index}.")

        image = self.transform(self.data[index]['front']['rgb']).float()
        seg = self.transform(self.data[index]['front']['seg']).squeeze().long()
        # images.shape = [3,128,128]

        return {'input': image, 'target': seg}

    def __len__(self):
        return self.length


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = r"D:\Code\CV_project\lstm_fast_fcn\dataset.pt"
    cd = CustomDataset(device, path)
    print(cd[0])
    # print(cd[0]['front']['seg'].shape)
