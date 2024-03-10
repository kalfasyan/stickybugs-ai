from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm.auto import tqdm

def test_model(model, loader):
    """Test a model on a dataloader."""

    model.eval()
    correct = 0
    y_pred,y_true = [],[]

    feats = ['imgname','platename','filename','plate_idx','location','date','year','xtra','width','height']
    info = {i:[] for i in feats}    
    for x_batch,y_batch,imgname,platename,filename,plate_idx,location,date,year,xtra,width,height in tqdm(loader, 
                                                                                                          desc='Testing..\t'):
        y_batch = torch.as_tensor(y_batch).type(torch.LongTensor)
        x_batch,y_batch = x_batch.cuda(), y_batch.cuda()
        pred = model(x_batch)
        _, preds = torch.max(pred, 1)
        y_pred.extend(preds.detach().cpu().numpy())
        y_true.extend(y_batch.detach().cpu().numpy())
        correct += (pred.argmax(axis=1) == y_batch).float().sum().item()

        info['imgname'].extend(imgname)
        info['platename'].extend(platename)
        info['filename'].extend(filename)
        info['plate_idx'].extend(plate_idx)
        info['location'].extend(location)
        info['date'].extend(date)
        info['year'].extend(year)
        info['xtra'].extend(xtra)
        info['width'].extend(width)
        info['height'].extend(height)

    bacc = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
    conf_mat = confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true')

    return bacc, conf_mat, y_true, y_pred, info

def get_gt_and_preds(model, loader, device, disable=False):
    """Get ground truth and predictions from a model and a dataloader."""

    model.eval()

    y_pred,y_true = [],[]
    for x_batch,y_batch,_,_,_,_,_,_,_,_,_,_ in tqdm(loader, 
                                                    desc='Getting ground truth and predictions..', 
                                                    total=len(loader),
                                                    disable=disable):
        y_batch = torch.as_tensor(y_batch).type(torch.LongTensor)
        x_batch,y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        _, preds = torch.max(pred, 1)
        y_pred.extend(preds.detach().cpu().numpy())
        y_true.extend(y_batch.detach().cpu().numpy())
    return y_true, y_pred

def plot_confusion_matrix(model, dataloader, dataset, labelticks, savename='', device='cuda'):
    """Plot a confusion matrix from a model and a dataloader."""

    y_true, y_pred = get_gt_and_preds(model, dataloader, device=device)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
    bacc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)

    plt.figure(figsize=(14, 10))
    sns.heatmap(conf_mat, annot=True, fmt='.1%', cmap='YlGnBu',
                xticklabels=labelticks, yticklabels=labelticks)
    plt.title(f"Balanced accuracy score: {bacc*100:.1f}%")
    plt.tight_layout()
    if savename:
        plt.savefig(f"{savename}")
