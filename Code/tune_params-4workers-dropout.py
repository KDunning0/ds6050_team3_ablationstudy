# %%
import os
# Fix PyTorch memory fragmentation (recommended in your OOM error)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import optuna
from optuna.trial import TrialState
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# As instructed, logging to weights and biases
import wandb
from model import SkinEffnetB4
# We'll probably need to change this call depending on the dataloader setup.
from dataloader import make_loaders, set_up
# This is how we'll avoid having 8 different files.
import argparse
# This gives us the option of using recall score instead of accuracy.
from sklearn.metrics import recall_score, roc_auc_score
import gc

# %%
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
# These are unused but kept in case desired.
use_EM = "EM" in arg.condition
use_META = "META" in arg.condition
use_feat_ext = "FEAT" in arg.condition

# %% [markdown]
# #### Examples for slurm call:
# 
# Transfer learning and nothing else:
# 
# python tune_params.py -c TL
# 
# Base with EM and Meta:
# 
# python tune_params.py -c EM_META

# %%
# or any other number.
SEED = 42
DEVICE, base_ds, train_idx, val_idx = set_up(SEED)
# I set this in the middle of where he wants it.
EPOCHS = 30

# %%
def objective(trial):

    # --- MEMORY FIX 1: Clear GPU cache before each trial ---
    gc.collect()
    torch.cuda.empty_cache()

    # instantiating WandB
    track = "TL" if use_TL else "SCRATCH"
    project_name = "ds6050_b4_baseline_tune_dropout_SCRATCH" if not use_TL else "ds6050_b4_baseline_tune_dropout_TL"
    wandb.init(entity="ds6050_team3", project=project_name, name=f"{track}_trial_{trial.number}", reinit='finish_previous')

    # We're calling in the model as shown in the pytorch_simple tutorial file but modified for our model.
    # The arguments are inputted for argparse and are controlled in the slurm calls.
    dropout_p = 0.5 if not use_TL else 0.0
    model = SkinEffnetB4(pretrained=use_TL, use_metablock=use_META, 
                     dropout_p=dropout_p, feature_extract=use_feat_ext).to(DEVICE)

    # Setting up the parameters we want to know about with reasonable ranges.
    # Learning rates usually pretty tiny for the skin data.
    # Incorporating overleaf feedback from the Professor.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wc = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    scheduler = trial.suggest_categorical("scheduler", ["cosine", "step"])

    # We're not testing optimizers.  We decided on AdamW.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wc)

    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    else:
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    CEloss = nn.CrossEntropyLoss()

    # --- MEMORY FIX 2: Initialize GradScaler for mixed precision (AMP) ---
    scaler = torch.amp.GradScaler('cuda')

    # Setting up dataloader
    batch_size = trial.suggest_categorical("batch_size", [16, 24, 32, 48, 64])

    # This can be changed when we have the dataloader file.
    train_load, valid_load = make_loaders(base_ds, train_idx, val_idx, 
                                          batch_size=batch_size, seed=SEED,
                                          use_equilibration = use_EM, num_workers=4)

    # Log hyperparameters to WandB so they appear alongside metrics in the dashboard
    wandb.config.update({
        "lr": lr,
        "weight_decay": wc,
        "scheduler": scheduler,
        "batch_size": batch_size,
        "dropout_p": dropout_p
    })
    
    # Doing the training:
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        # Throws away image IDs since we don't need them here.
        for batch_idx, (data, target, meta_data, _) in enumerate(train_load):
            # We don't need to flatten it like in the tutorial code.
            data, target = data.to(DEVICE), target.to(DEVICE)
            if meta_data is not None:
                meta_data = meta_data.to(DEVICE)

            optimizer.zero_grad()

            # --- MEMORY FIX 3: Wrap forward pass in autocast for mixed precision ---
            # This cuts memory usage ~40-50% by using float16 where safe.
            with torch.amp.autocast('cuda'):
                output = model(data, meta_data) if use_META else model(data)
                loss = CEloss(output, target)

            # --- MEMORY FIX 3 cont: Use scaler instead of loss.backward() + optimizer.step() ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
        train_loss = running_loss / len(train_load)

        # Doing the Validation
        # Crafting for mean-recall instead of accuracy due to imbalance.
        model.eval()
        # For mean recall/Weighted average
        preds = []
        # For ROC AUC
        probs = []
        labs = []
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target, meta_data, _) in enumerate(valid_load):
                data, target = data.to(DEVICE), target.to(DEVICE)
                if meta_data is not None:
                    meta_data = meta_data.to(DEVICE)
                # --- MEMORY FIX 4: autocast during validation too ---
                with torch.amp.autocast('cuda'):
                    output = model(data, meta_data) if use_META else model(data)
                    val_loss_batch = CEloss(output, target)
                val_running_loss += val_loss_batch.item()
                # Removed keep dim for argmax.
                pred = output.argmax(dim=1)
                # --- PROB FIX 1: Cast back to float32 after autocast, then renormalize
                # at the tensor level so rows sum to exactly 1.0 before leaving the GPU.
                # autocast can introduce float16 precision errors that cause sklearn's
                # strict probability sum check to fail.
                prob = F.softmax(output, dim=1).float()
                prob = prob / prob.sum(dim=1, keepdim=True)
                preds.extend(pred.cpu().numpy())
                probs.extend(prob.cpu().numpy())
                labs.extend(target.cpu().numpy())

        val_loss = val_running_loss / len(valid_load)

        # Calculating ROC AUC
        probs_np = np.array(probs)
        # --- PROB FIX 2: Final numpy-level renormalization as a safety net right
        # before sklearn sees the data, catching any remaining floating point drift.
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
        auc_by_class = roc_auc_score(labs, probs_np, multi_class='ovr', average=None)
        
        # This is basically balanced accuracy.        
        m_recall = recall_score(labs, preds, average='macro')

        # Logging to Weights and Biases
        wandb.log({
            "m_recall": m_recall,
            "mean_auc": auc_by_class.mean(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "auc_class_0_MEL": auc_by_class[0],
            "auc_class_1_NV": auc_by_class[1],
            "auc_class_2_BCC": auc_by_class[2],
            "auc_class_3_AK": auc_by_class[3],
            "auc_class_4_BKL": auc_by_class[4],
            "auc_class_5_DF": auc_by_class[5],
            "auc_class_6_VASC": auc_by_class[6],
            "auc_class_7_SCC": auc_by_class[7],
            "epoch": epoch
        })

        sched.step()

        trial.report(m_recall, epoch)

        if trial.should_prune():
            wandb.finish()
            del model, optimizer, sched, train_load, valid_load, scaler
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        
    wandb.finish()
    del model, optimizer, sched, train_load, valid_load, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return m_recall

# Almost unchanged from pytorch_simple tutorial.  Changed n_trials number and removed timeout.
if __name__ == "__main__":

    # Add SQL lite database to carry over for use in runner.
    # Since we're only doing two studies: one for TL and one for Scratch, we handle it this way.
    if use_TL:
        study = optuna.create_study(study_name="TL_DO",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                storage=f"sqlite:///optuna_TL_dropout.db",
                                load_if_exists=True, 
                                direction="maximize")
    else:
        study = optuna.create_study(study_name="SCRATCH_DO",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                storage=f"sqlite:///optuna_SCRATCH_dropout.db",
                                load_if_exists=True, 
                                direction="maximize")
        
    study.optimize(objective, n_trials=40, catch=(torch.OutOfMemoryError, ValueError))

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

# %%
# Extra Citations
# 1. Mbambo, T. (n.d.). Argparse tutorial. Python documentation. https://docs.python.org/3/howto/argparse.html 
# 2. Rosebrock, A. (2018, March 12). Python argparse, and command line arguments. 
# PyImageSearch. https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/ 
# 3. Ramos, L. (n.d.). Build command-line interfaces with python's argparse. 
# Real Python. https://realpython.com/command-line-interfaces-python-argparse/ 
# 4. Optuna Contributors. (2018). Tutorial. Tutorial - Optuna 4.7.0 documentation. 
# https://optuna.readthedocs.io/en/stable/tutorial/index.html 
# 5. Wadekar, S. (2021, January 19). Optuna: Hyperparameter optimization in pytorch . 
# Medium. https://medium.com/swlh/optuna-hyperparameter-optimization-in-pytorch-9ab5a5a39e77