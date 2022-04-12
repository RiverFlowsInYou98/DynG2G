import argparse
import yaml
import os
import time
import torch
import torch.nn
import numpy as np
import copy
import pickle

from utils import *
from models import *
from data import *

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "-f",
    "--config_file",
    default="configs/config-1.yaml",
    help="Configuration file to load.",
)
ARGS = parser.parse_args(args=[])

with open(ARGS.config_file, "r") as f:
    config = yaml.safe_load(f)
print("Loaded configuration file", ARGS.config_file)
print(config)

K = config["K"]
n_hidden = config["n_hidden"]
tolerance = config["tolerance"]
L_list = config["L_list"]
save_MRR_MAP = config["save_MRR_MAP"]
save_sigma_mu = config["save_sigma_mu"]

p_val = config["p_val"]
p_test = config["p_test"]
p_train = 1 - p_val - p_test

scale = False
verbose = True

data_name = "SBM"
name = "Results/" + data_name + "_" + os.environ["SLURM_ARRAY_JOB_ID"] + "/"
logger = init_logging_handler(name)
logger.debug(str(config))

device = check_if_gpu()
logger.debug("The code will be running on {}.".format(device))

print("Dataset: " + str(data_name))
if data_name == "SBM":
    data = Dataset_SBM("datasets/sbm_50t_1000n_adj.csv")
    L = 64
    num_epochs = 700
    patience_init = 10

elif data_name == "UCI":
    data = Dataset_UCI(
        (
            "datasets/download.tsv.opsahl-ucsocial.tar.bz2",
            "opsahl-ucsocial/out.opsahl-ucsocial",
        )
    )
    L = 256
    num_epochs = 100
    patience_init = 3

# data.A_list = [data.A_list[0]] * len(data)
# data.X_list = [data.X_list[0]] * len(data)


learning_rate = 1e-3
train_time_list = []
mu_list = []
sigma_list = []
theta = 0.25
resetting_counts = 0


def Validate_onLinkPredScore(A, mu, sigma):
    """
    use link prediction score to validate the embedding
    """
    L = mu.shape[1]
    num_samples = min(1000, A.nnz)
    val_edges = np.row_stack(
        (sample_ones(A, num_samples), sample_zeros(A, num_samples))
    )
    neg_val_energy = -Energy_KL(mu, sigma, val_edges, L)
    val_label = A[val_edges[:, 0], val_edges[:, 1]].A1
    val_auc, val_ap = score_link_prediction(
        val_label, neg_val_energy.cpu().detach().numpy()
    )
    return val_auc, val_ap


for t in range(len(data)):
    logger.debug("timestamp {}".format(t))
    A, X = data[t]
    N, D = X.shape
    hops = get_hops(A, K)
    scale_terms = {}
    for h in hops:
        if h != -1:
            scale_terms[h] = hops[h].sum(1).A1
        else:
            scale_terms[max(hops.keys()) + 1] = hops[1].shape[0] - hops[h].sum(1).A1

    start = time.time()
    if t == 0:
        G2G = Graph2Gauss(n_hidden, L, D)
    elif t == 1:
        G2G = copy.deepcopy(G2G)
        G2G_prev = copy.deepcopy(G2G)
        if G2G.layers[0].weight.data.shape[0] < D:
            dummy_input = InputLinear(G2G.layers[0].weight.data.shape[0])
            dummy_output, G2G.layers[0] = wider(dummy_input, G2G.layers[0], D)
    else:
        G2G = copy.deepcopy(G2G)
        G2G_prev2 = copy.deepcopy(G2G_prev)
        G2G_prev = copy.deepcopy(G2G)

        add_net1 = copy.deepcopy(G2G_prev)
        add_net2 = copy.deepcopy(G2G_prev2)
        if G2G.layers[0].weight.data.shape[0] < D:
            dummy_input = InputLinear(G2G.layers[0].weight.data.shape[0])
            dummy_output, G2G.layers[0] = wider(dummy_input, G2G.layers[0], D)
        if add_net1.layers[0].weight.data.shape[0] < D:
            dummy_input = InputLinear(add_net1.layers[0].weight.data.shape[0])
            dummy_output, add_net1.layers[0] = wider(dummy_input, add_net1.layers[0], D)
        if add_net2.layers[0].weight.data.shape[0] < D:
            dummy_input = InputLinear(add_net2.layers[0].weight.data.shape[0])
            dummy_output, add_net2.layers[0] = wider(dummy_input, add_net2.layers[0], D)
        for param1, param2, param3 in zip(
            G2G.parameters(), add_net1.parameters(), add_net2.parameters()
        ):
            param1.data = theta * param2.data + (1 - theta) * param3.data

    G2G = G2G.to(device)
    optimizer = torch.optim.Adam(G2G.parameters(), lr=learning_rate)
    patience = patience_init
    wait = 0
    best = 0

    if t < len(data):
        logger.debug("Training")
        G2G.train()
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            X = X.to(device)
            _, mu, sigma = G2G(X)
            triplets, triplet_scale_terms = to_triplets(
                sample_all_hops(hops), scale_terms
            )
            loss_s = build_loss(
                triplets, triplet_scale_terms, mu, sigma, L, scale=scale
            )
            if loss_s > 1e4:
                logger.debug("Loss overflow, resetting G2G model")
                resetting_counts = resetting_counts + 1
                G2G.reset_parameters()
                _, mu, sigma = G2G(X)
                triplets, triplet_scale_terms = to_triplets(
                    sample_all_hops(hops), scale_terms
                )
                loss_s = build_loss(
                    triplets, triplet_scale_terms, mu, sigma, L, scale=scale
                )
            loss_s.backward()
            optimizer.step()
            if verbose and (epoch == 1 or epoch % 10 == 0):
                val_auc, val_ap = Validate_onLinkPredScore(A, mu, sigma)
                logger.debug(
                    "L: {}, epoch: {:3d}, loss: {:.4f}, val_auc: {:.4f}, val_ap: {:.4f}".format(
                        L, epoch, loss_s.item(), val_auc, val_ap
                    )
                )
                wait += 1
                if val_auc + val_ap > best:
                    best = val_auc + val_ap
                    wait = 0
                if wait >= patience:
                    logger.debug("L: {}, epoch: {:3d}: Early Stopping".format(L, epoch))
                    break
    else:
        logger.debug("Testing")
        G2G.eval()
        X = X.to(device)
        _, mu, sigma = G2G(X)

    end = time.time()
    train_time_list.append(end - start)

    mu_list.append(mu.cpu().detach().numpy())
    sigma_list.append(sigma.cpu().detach().numpy())
    val_auc, val_ap = Validate_onLinkPredScore(A, mu, sigma)
    logger.debug("val_auc {:.4f}, val_ap: {:.4f}".format(val_auc, val_ap))

print("Training finished!")
print(
    "G2G model resets %d times in %d time stamps during training."
    % (resetting_counts, len(data))
)
if save_sigma_mu == True:
    if not os.path.exists(name + "/saved_embed"):
        os.makedirs(name + "/saved_embed")
    with open(name + "/saved_embed/mu" + str(L), "wb") as f:
        pickle.dump(mu_list, f)
    with open(name + "/saved_embed/sigma" + str(L), "wb") as f:
        pickle.dump(sigma_list, f)
