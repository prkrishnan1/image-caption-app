from pycocotools.coco import COCO
import torch
import numpy as np
import matplotlib.pyplot as plt
import pylab
import json
import os
import pandas as pd
from datasets import Dataset, Image
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from PIL import Image as PILImage
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(
                    prog='train_image_captioning_model',
                    description='Loads data and runs training and evaluation for microsoft/git-base model')

parser.add_argument('-w', '--workdir')

class ImageCaptioningDataset(TorchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding

def load_image_captioning_dataset(split: str, workdir: str='/workspace/') -> Dataset:
    dataDir = '{}/cocodatasetnew/cocodataset'.format(workdir)
    dataType = '{}2017'.format(split)
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    
    # Initialize coco API by passing in captions JSON
    coco_caps=COCO(annFile)

    # Form a set of file paths for training images from image metadata
    with open(annFile, 'r') as fp:  
        image_metadata = json.load(fp)

    image_ids = [image_entry["id"] for image_entry in image_metadata["images"]]
    image_file_paths = ['{}/{}/{}'.format(dataDir, dataType, image_entry["file_name"]) for image_entry in image_metadata["images"]]
    
    id_to_file_path = dict(zip(image_ids, image_file_paths))

    # Load all captions from coco API
    image_captions_all = coco_caps.loadAnns(coco_caps.getAnnIds(imgIds=image_ids))
    image_captions_unique = {}

    # Form a 1:1 image id to caption mapping
    for caption in image_captions_all:
        if not caption["image_id"] in image_captions_unique:
            image_captions_unique[caption["image_id"]] = caption["caption"]
    
    dataset_loaded = Dataset.from_dict({"image": [id_to_file_path[key] for key in image_ids], "text": [image_captions_unique[key] for key in image_ids]}).cast_column("image", Image())

    return dataset_loaded
    
def train_model(train_dataset: Dataset, model, processor, device):
    epochs=10
    learning_rate=5e-5
    batch_size=16

    train_dataset = ImageCaptioningDataset(dataset=train_dataset, processor=processor)
    train_dataloader = TorchDataLoader(train_dataset, shuffle=True, batch_size=batch_size)

#     batch = next(iter(train_dataloader))
    
#     for k,v in batch.items():
#         print(k,v.shape)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("Device Used --->", device)
    model.to(device)
    
    model.train()
    
    for epoch in range(epochs):
        print("Epoch:", epoch)
        for idx, batch in tqdm(enumerate(train_dataloader)):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)

            loss = outputs.loss

            print("Loss:", loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    print("Finished training!")
    
def evaluate_model(eval_dataset : Dataset, model=AutoModelForCausalLM.from_pretrained("microsoft/git-base"), processor=AutoProcessor.from_pretrained("microsoft/git-base"), device='cuda'):
    captions = []
    print("Running model on evaluation dataset")
    
    for i in range(len(eval_dataset)):
        image = eval_dataset[i]["image"]
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        captions.append(generated_caption)
    
    with open("generated_captions_outfile.txt", "w") as fp:
        json.dump(captions, fp)
    
    print("Wrote captions to outfile!")

if __name__ == '__main__':
    args = parser.parse_args()

    print("Step 1: Data Loading")
    # train_dataset = load_image_captioning_dataset('train', args.workdir)

    print("Step 2: Loaded Dataset. Beginning Model Training")
    # Define training objects
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # train_model(train_dataset, model, processor, device)

    print("Step 3: Beginning Model evaluation")
    # eval_dataset = load_image_captioning_dataset('val', args.workdir)
    # evaluate_model(eval_dataset, model, processor, device)
    
    print("Step 4: Saving Model")
    
    processor.save_pretrained('{0}/cocodatasetnew/cocodataset/tokenizers/imagecaptioning/2/'.format(args.workdir))
    
    model.save_pretrained('{0}/cocodatasetnew/cocodataset/models/imagecaptioning/2/'.format(args.workdir))

    print("Done!")
