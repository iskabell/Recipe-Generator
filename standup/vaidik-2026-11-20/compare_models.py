"""Vaidik's frozen-feature ResNet18 vs YOLO26n-cls experiment.

Run from repository root. This trains fresh heads; it does not reuse a teammate's
trained classifier. Model-specific input normalization is recorded in metrics.
No bounding-box detection results are claimed.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'Data/grocery_store/processed'


def make_splits(rows, seed=42):
    by_class = defaultdict(list)
    for i, row in enumerate(rows):
        if row['split'] == 'train':
            by_class[int(row['label'])].append(i)
    rng = random.Random(seed)
    train, dev = [], []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        n = max(1, math.floor(len(indices) * .15)) if len(indices) >= 2 else 0
        dev.extend(indices[:n])
        train.extend(indices[n:])
    test = [i for i, r in enumerate(rows) if r['split'] == 'test']
    splits = {'train': train, 'dev': dev, 'test': test}
    assert not (set(train) & set(dev) or set(train) & set(test) or set(dev) & set(test))
    for split, ids in splits.items():
        assert len({int(rows[i]['label']) for i in ids}) == 43, split
    return splits


class Images(Dataset):
    def __init__(self, rows, name):
        self.rows = rows
        ops = [transforms.ToTensor()]
        if name == 'resnet18':
            ops.append(transforms.Normalize([.485, .456, .406], [.229, .224, .225]))
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(DATA / self.rows[i]['image_path']) as im:
            return self.transform(im.convert('RGB'))


def backbone(name):
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        from ultralytics import YOLO
        model = YOLO('yolo26n-cls.pt').model
        dim = model.model[-1].linear.in_features
        model.model[-1].linear = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dim


def embeddings(model, x):
    out = model(x)
    if isinstance(out, tuple):
        out = out[1]  # YOLO eval returns (softmax, pre-softmax features).
    assert isinstance(out, torch.Tensor) and out.ndim == 2
    return out


def metrics(y, logits):
    predicted = logits.argmax(1).numpy()
    truth = y.numpy()
    return {'top1_accuracy': float(accuracy_score(truth, predicted)),
            'top5_accuracy': float((logits.topk(5, dim=1).indices == y[:, None]).any(1).float().mean()),
            'macro_f1': float(f1_score(truth, predicted, labels=list(range(43)), average='macro', zero_division=0)),
            'per_class_recall': recall_score(truth, predicted, labels=list(range(43)), average=None, zero_division=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['resnet18', 'yolo26n'], default=['resnet18', 'yolo26n'])
    parser.add_argument('--resume', action='store_true', help='Reuse completed models with identical split and settings')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    with (DATA/'manifest.csv').open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['split'] != 'validation']
    with (DATA/'labels.csv').open() as f:
        labels = list(csv.DictReader(f))
    splits = make_splits(rows)
    output = HERE/'results'; output.mkdir(exist_ok=True)
    split_records = [{'image_path': rows[i]['image_path'], 'split': s, 'label': int(rows[i]['label'])}
                     for s, ids in splits.items() for i in ids]
    with (output/'experiment_splits.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(split_records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(split_records)
    fingerprint = hashlib.sha256(json.dumps(split_records, sort_keys=True).encode()).hexdigest()
    report = {'author': 'Vaidik Nadheria', 'seed':42, 'device':'cpu',
              'method':'Frozen pretrained feature extractors; independently trained linear heads',
              'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
              'dataset_manifest_sha256':hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest(),
              'split_sha256':fingerprint,'split_sizes':{s:len(ids) for s,ids in splits.items()},
              'class_names':[r['ingredient'] for r in labels],
              'hyperparameters':{'epochs':args.epochs,'optimizer':'AdamW','lr':.003,'weight_decay':.0001,'batch_size':256},
              'limitations':['Random image-level development split can share related scenes with training.',
                             'Single-source grocery photos do not establish performance on kitchens or multi-item scenes.',
                             'Only classifier heads trained; no full-backbone fine-tuning or object detection.'],
              'models':{}}
    if (output/'metrics.json').exists():
        previous = json.loads((output/'metrics.json').read_text())
        if not args.resume:
            raise FileExistsError('Results exist. Use --resume to reuse completed model runs.')
        if previous['split_sha256'] != fingerprint or previous['hyperparameters'] != report['hyperparameters']:
            raise ValueError('Cannot resume different splits or training settings')
        report['models'] = previous['models']
    y=torch.tensor([int(r['label']) for r in rows], dtype=torch.long)
    for name in args.models:
        if name in report['models']:
            print(f'Reusing completed {name} result', flush=True)
            continue
        print(f'Loading {name}',flush=True)
        started=time.perf_counter()
        model,dim=backbone(name)
        parts=[]
        loader=DataLoader(Images(rows,name),batch_size=args.batch_size,shuffle=False,num_workers=0)
        with torch.inference_mode():
            for n,x in enumerate(loader):
                parts.append(embeddings(model,x).detach().clone())
                if n%20==0: print(f'{name}: feature batch {n+1}/{len(loader)}',flush=True)
        features=torch.cat(parts)
        feature_seconds=time.perf_counter()-started
        torch.manual_seed(42)
        head=nn.Linear(dim,43)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.0001)
        criterion=nn.CrossEntropyLoss()
        best_f1=-1; best_epoch=0; best=None; history=[]
        train_ids=torch.tensor(splits['train']);dev_ids=torch.tensor(splits['dev']);test_ids=torch.tensor(splits['test'])
        for epoch in range(1,args.epochs+1):
            head.train()
            order=train_ids[torch.randperm(len(train_ids))]
            for ids in order.split(256):
                optimizer.zero_grad();loss=criterion(head(features[ids]),y[ids]);loss.backward();optimizer.step()
            head.eval()
            with torch.no_grad():
                dev=metrics(y[dev_ids],head(features[dev_ids]))
            history.append({'epoch':epoch,'dev_accuracy':dev['top1_accuracy'],'dev_macro_f1':dev['macro_f1']})
            if dev['macro_f1']>best_f1:
                best_f1=dev['macro_f1']; best_epoch=epoch; best=copy.deepcopy(head.state_dict())
        head.load_state_dict(best);head.eval()
        with torch.no_grad():
            test_logits=head(features[test_ids]);test=metrics(y[test_ids],test_logits)
        # Batch-1 forward latency: same machine, excludes disk loading and UI.
        sample=Images(rows,name)[splits['test'][0]].unsqueeze(0)
        timings=[]
        with torch.inference_mode():
            for _ in range(5):head(embeddings(model,sample))
            for _ in range(30):
                start=time.perf_counter();head(embeddings(model,sample));timings.append(1000*(time.perf_counter()-start))
        torch.save({'model':name,'head':best,'feature_dim':dim,'class_names':report['class_names'],
                    'split_sha256':fingerprint},output/f'{name}_head.pt')
        matrix=confusion_matrix(y[test_ids].numpy(),test_logits.argmax(1).numpy(),labels=list(range(43)))
        np.savetxt(output/f'{name}_confusion.csv',matrix,delimiter=',',fmt='%d')
        test_rows=[]
        for i,logit in zip(splits['test'],test_logits):
            top=torch.topk(logit.softmax(0),3)
            test_rows.append({'image_path':rows[i]['image_path'],'true_label':int(rows[i]['label']),
                              'predicted_label':int(top.indices[0]),
                              'top3':[{'label':int(k),'probability':float(v)} for k,v in zip(top.indices,top.values)]})
        (output/f'{name}_predictions.json').write_text(json.dumps(test_rows,indent=2)+'\n')
        report['models'][name]={'best_epoch':best_epoch,'dev_macro_f1':best_f1,
                                 'test':test,'median_cpu_forward_ms':float(np.median(timings)),
                                 'feature_extraction_seconds':feature_seconds,'history':history,
                                 'normalization':'ImageNet mean/std' if name=='resnet18' else 'RGB / 255; no mean/std normalization'}
        (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,test,'latency',np.median(timings),flush=True)
        del model,features,parts
    print('Results:',output,flush=True)


if __name__=='__main__':
    main()
