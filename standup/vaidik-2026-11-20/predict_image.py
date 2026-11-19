"""Predict an individual ingredient with Vaidik's trained classifier head."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from compare_models import backbone, HERE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--model',choices=['resnet18','yolo26n'],default='yolo26n')
    args=parser.parse_args()
    torch.set_num_threads(4)
    checkpoint=torch.load(HERE/'results'/f'{args.model}_head.pt',map_location='cpu',weights_only=True)
    model,dim=backbone(args.model)
    head=torch.nn.Linear(dim,43)
    head.load_state_dict(checkpoint['head']);head.eval()
    with Image.open(args.image) as im:
        im=ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(224,224),method=Image.Resampling.LANCZOS,color=(128,128,128))
        pixels=transforms.ToTensor()(im)
    if args.model=='resnet18':
        pixels=transforms.Normalize([.485,.456,.406],[.229,.224,.225])(pixels)
    with torch.inference_mode():
        features=model(pixels.unsqueeze(0))
        if isinstance(features,tuple):features=features[1]
        values,indices=head(features).softmax(1)[0].topk(3)
    print(json.dumps([{'ingredient':checkpoint['class_names'][int(i)],'score':round(float(v),4)}
                      for i,v in zip(indices,values)],indent=2))
    print('Scores are model outputs, not calibrated certainty. Confirm the ingredient before recipe matching.')


if __name__=='__main__':
    main()
