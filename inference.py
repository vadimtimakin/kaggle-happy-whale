from catboost import train
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd
import cv2
from sklearn.neighbors import NearestNeighbors

from utils import *
from config import config
from objects.model import LesGoNet
from objects.transforms import SmartResize


class DolphinDataset(Dataset):
    """Dataset structure."""

    def __init__(self, config, images, is_train):
        super().__init__()

        self.is_train = is_train

        if is_train:
            path = config.paths.path_to_images
        else:
            path = config.paths.path_to_test

        self.img_paths = [os.path.join(path, i) for i in images]
        self.data_len = len(self.img_paths)

        self.transforms = A.Compose([
            A.Resize(384, 512),
            A.Normalize(),
        ])

        self.flip = A.HorizontalFlip(p=1.0)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        image = np.array(Image.open(img_path))
        if len(image.shape) != 3:
            image = np.stack((image, ) * 3, axis=-1)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)["image"]
        # image_flipped = self.flip(image=image)["image"]

        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).unsqueeze(0)

        # image_flipped = np.moveaxis(image_flipped, -1, 0)
        # image_flipped = torch.from_numpy(image_flipped).unsqueeze(0)

        # return image, image_flipped
        return image


def get_data_loader(config, is_train, images):
    """Gets a PyTorch Dataloader."""
    dataset = DolphinDataset(config, images, is_train)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        **config.data.dataloader_params,
    )
    return data_loader


def get_inference_loaders(config):
    """Get PyTorch Dataloaders."""

    train_df = pd.read_csv(config.paths.path_to_csv)
    train_images = train_df[config.data.id_column].values
    train_labels = train_df[config.data.target_column].values

    test_df = pd.read_csv(config.paths.path_to_sample_sub)
    test_images = test_df[config.data.id_column].values

    train_loader = get_data_loader(
        is_train=True,
        config=config,
        images=train_images,
    )

    val_loader = get_data_loader(
        is_train=False,
        config=config,
        images=test_images,
    )

    return train_loader, val_loader, train_labels


def extract_embeddings(config, models, dataloader):
    embeddings = []

    with torch.inference_mode():
        for inputs in tqdm(dataloader):
            preds = torch.zeros((inputs.shape[0], config.model_params.embedding_size))
            preds = preds.to(config.training.device)

            inputs = inputs.to(config.training.device)
            if len(inputs.shape) == 5:
                inputs = inputs.squeeze(1)

            for model in models:
                preds += model(inputs.float(), get_embeddings=True)
            
            preds /= len(models)
            embeddings.append(preds.cpu())

    return np.concatenate(embeddings)



def inference(config):
    '''Inference function.'''

    print('Starting inference')

    new_individual_thres = 0.5

    train_loader, test_loader, train_labels = get_inference_loaders(config)

    torch.cuda.empty_cache()

    model = LesGoNet(config.model_params)
    model.to(config.training.device)
    models = []
    for checkpoint in config.inference_checkpoints:
        model.load_state_dict(torch.load(checkpoint)["model"])
        model.eval()
        models.append(model)

    print("Extracting train embeddings")
    train_embeddings = extract_embeddings(config, models, train_loader)

    print("Extracting test embeddings")
    test_embeddings = extract_embeddings(config, models, test_loader)

    neigh = NearestNeighbors(n_neighbors=config.training.n_neighbors,metric="cosine")
    neigh.fit(train_embeddings)

    distances,idxs = neigh.kneighbors(test_embeddings, return_distance=True)
    conf = 1-distances
    preds=[]

    for j in range(len(idxs)):
        preds.append(list(train_labels[idxs[j]]))

    allTop5Preds=[]
    for i in range(len(preds)):

        predictTop = preds[i][:5]
        Top5Conf = conf[i][:5]

        if Top5Conf[0] < new_individual_thres:
           
            tempList=['new_individual',predictTop[0],predictTop[1],predictTop[2],predictTop[3]]
            allTop5Preds.append(tempList)   
           
        elif Top5Conf[1] < new_individual_thres:
   
            tempList=[predictTop[0],'new_individual',predictTop[1],predictTop[2],predictTop[3]]
            allTop5Preds.append(tempList)    
           
        elif Top5Conf[2] < new_individual_thres:

            tempList=[predictTop[0],predictTop[1],'new_individual',predictTop[2],predictTop[3]]
            allTop5Preds.append(tempList)    
           
        elif Top5Conf[3] < new_individual_thres:
           
            tempList=[predictTop[0],predictTop[1],predictTop[2],'new_individual',predictTop[3]]        
            allTop5Preds.append(tempList)  
           
        elif Top5Conf[4] < new_individual_thres:

            tempList=[predictTop[0],predictTop[1],predictTop[2],predictTop[3],'new_individual']        
            allTop5Preds.append(tempList)        
           
        else:
            allTop5Preds.append(predictTop)

    allTop5Preds = [' '.join(i) for i in allTop5Preds]

    submission = pd.read_csv(config.paths.path_to_sample_sub)
    submission['predictions'] = allTop5Preds
    submission.to_csv('submission.csv', index=False)
    print(submission.head())

if __name__ == "__main__":
    inference(config)
