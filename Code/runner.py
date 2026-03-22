import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from isic2019_dataset import get_test_dataset
from model import SkinEffnetB4
from dataloader import set_up,make_loaders
import optuna
# This is how we'll avoid having 8 different files.
import argparse
# This gives us the option of using recall score instead of accuracy.
from sklearn.metrics import recall_score, roc_auc_score
import wandb
import copy

# IMPORTANT: Download the dataset via the instructions in isic2019_dataset.py before running runner.py.

# We're going to use these as switches to call slurm for parallelization.
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--condition", 
                required=True, 
                type=str,
                help='''
                The condition to train on.
                TL : Transfer Learning
                EM : Equilibrium Minibatch
                META : Metablock
                FEAT: Optional frozen backbone
                ''')
arg = ap.parse_args()

# This is also for the slurm calls.
use_TL = "TL" in arg.condition
use_EM = "EM" in arg.condition
use_META = "META" in arg.condition
use_feat_ext = "FEAT" in arg.condition

def evaluate(model, dataloader, criterion, device, use_META = use_META):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    preds = []
    # For ROC AUC
    probs = []
    labs = []
    with torch.no_grad():
        for batch_idx, (data, target, meta_data, _) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            if meta_data is not None:
                meta_data = meta_data.to(device)
            output = model(data, meta_data) if use_META else model(data)

            loss = criterion(output, target)
            running_loss += loss.item()
            
            # Removed keep dim for argmax.
            pred = output.argmax(dim=1)
            prob = F.softmax(output, dim=1)
            preds.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())
            labs.extend(target.cpu().numpy())

    # Calculating ROC AUC
    probs_np = np.array(probs)
    auc_by_class = roc_auc_score(labs, probs_np, multi_class='ovr', average=None)
        
    # This is basically balanced accuracy.        
    m_recall = recall_score(labs, preds, average='macro')

    epoch_loss = running_loss / len(dataloader)
    
    return epoch_loss, m_recall, auc_by_class, preds, labs


def _set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def train_epoch(model, dataloader, criterion, optimizer, device, feature_extract = use_feat_ext, use_META = use_META):
    """Train the model for one epoch."""
    model.train()
    if feature_extract:
        model.apply(_set_bn_eval)
    running_loss = 0.0

    for batch_idx, (data, target, meta_data, _) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        if meta_data is not None:
            meta_data = meta_data.to(device)

        optimizer.zero_grad()
        outputs = model(data, meta_data) if use_META else model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def train_model(device, model, train_loader, val_loader, lr, weight_decay, 
                scheduler, num_epochs=30, feature_extract = use_feat_ext, use_META = use_META):
    """
    Train and evaluate a model.

    Returns:
        the model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=lr)
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    else:
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # For early stopping
    best_m_recall = 0.0
    patience_count = 0
    # The number of epochs to wait before early stopping.  Can change.
    patience = 10
    # Initializing best weights incase there is never an improvement
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, feature_extract = use_feat_ext, use_META = use_META
        )
        epoch_loss, m_recall, auc_by_class, _, _ = evaluate(model, val_loader, criterion, device, use_META=use_META)

        wandb.log({
        "train_loss": train_loss,
        "val_loss": epoch_loss,
        "val_m_recall": m_recall,
        "val_mean_auc": auc_by_class.mean(),
        "val_auc_class_0_MEL": auc_by_class[0],
        "val_auc_class_1_NV": auc_by_class[1],
        "val_auc_class_2_BCC": auc_by_class[2],
        "val_auc_class_3_AK": auc_by_class[3],
        "val_auc_class_4_BKL": auc_by_class[4],
        "val_auc_class_5_DF": auc_by_class[5],
        "val_auc_class_6_VASC": auc_by_class[6],
        "val_auc_class_7_SCC": auc_by_class[7],
        "epoch": epoch
        })
        
        # Early stopping mechanism
        if m_recall > best_m_recall:
            best_m_recall = m_recall
            patience_count = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        sched.step()

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"{arg.condition}_best_weights.pth")

    return model


def main() -> None:
    SEED = 42

    # Getting data and also setting up the device.
    device, base_ds, train_idx, val_idx = set_up(SEED)
    test_ds = get_test_dataset()

    # Getting hyperparameters
    if use_TL:
        study = optuna.load_study(study_name="TL",
                                  storage=f"sqlite:///optuna_TL.db")

    else:
        study = optuna.load_study(study_name="SCRATCH_DO_v2",
                              storage=f"sqlite:///optuna_SCRATCH_dropout_v2.db")
    best_params = study.best_params

    # Dataloader
    train_load, val_load = make_loaders(base_ds, train_idx, val_idx,
                                        batch_size = best_params["batch_size"],
                                        seed=SEED,
                                        use_equilibration = use_EM,
                                        num_workers=2)
    test_load = DataLoader(test_ds,
                           batch_size=best_params["batch_size"],
                           shuffle=False,
                           num_workers=2)
    
    # Instantiating the Model
    dropout_p = 0.0 if use_TL else 0.5
    model = SkinEffnetB4(pretrained = use_TL, use_metablock = use_META, feature_extract = use_feat_ext, dropout_p=dropout_p)
    
    # Initializing W&B
    wandb.init(project="ds6050-g03-ISIC2019-Experiments", name=arg.condition)

    wandb.config.update({
    "lr": best_params["lr"],
    "weight_decay": best_params["weight_decay"],
    "scheduler": best_params["scheduler"],
    "batch_size": best_params["batch_size"],
    "use_TL": use_TL,
    "use_EM": use_EM,
    "use_META": use_META,
    "condition": arg.condition,
    'dropout_p': dropout_p
    })

    # Training
    model = train_model(device, model, train_load, val_load, 
                        lr=best_params["lr"], weight_decay=best_params["weight_decay"], 
                        scheduler=best_params["scheduler"],
                        num_epochs=30)
    
    # Testing
    criterion = nn.CrossEntropyLoss()
    test_loss, test_recall, test_auc, test_preds, test_labs = evaluate(model, test_load, criterion, device, use_META=use_META)

    # Logging metrics
    wandb.log({
        "test_loss": test_loss,
        "test_recall": test_recall,
        "test_mean_auc": test_auc.mean(),
        "test_auc_class_0_MEL": test_auc[0],
        "test_auc_class_1_NV": test_auc[1],
        "test_auc_class_2_BCC": test_auc[2],
        "test_auc_class_3_AK": test_auc[3],
        "test_auc_class_4_BKL": test_auc[4],
        "test_auc_class_5_DF": test_auc[5],
        "test_auc_class_6_VASC": test_auc[6],
        "test_auc_class_7_SCC": test_auc[7],
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_labs,
            preds=test_preds,
            class_names=["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
        )
    })

    # Closing W&B
    wandb.finish()

if __name__ == '__main__':
    main()