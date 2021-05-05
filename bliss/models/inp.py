from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Bernoulli, Normal
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from sklearn.cluster import KMeans
from pytorch_lightning import LightningModule

from ..datasets.sdss import StarStamper
from ..utils import SequentialVarg, SplitLayer, NormalEncoder, MLP, ConcatLayer

class HNP(nn.Module):
    """
    This is an implementation of the Hierarchical Neural Process (HNP), a new model.
    """

    def __init__(
        self,
        dep_graph,
        z_inference,
        z_pooler,
        h_prior,
        h_pooler,
        y_decoder,
        fb_z=None,
    ):
        super().__init__()
        ## Learned Submodules
        self.dep_graph = dep_graph
        self.z_inference = z_inference
        self.z_pooler = z_pooler
        self.h_prior = h_prior
        self.h_pooler = h_pooler
        self.y_decoder = y_decoder
        ## Initialize free-bits regularization
        self.fb_z = fb_z

    def encode(self, X, S):
        n_inputs = S.size(0)

        ## Calculate dependency graph
        G = self.dep_graph(X)

        ## Calculate the prior distribution for the H
        pH = self.h_prior(X, G)

        if n_inputs > 0:
            ## Encode the available stamps
            Zi = self.z_inference(X[:n_inputs], S)
            ## Sample the hierarchical latent variables from the latent variables
            # qH = self.h_pooler(X, G, Zi)
            # qH = self.h_pooler(X, Zi, G[:n_inputs].transpose(1, 0))
            qH = self.h_pooler(X, Zi, G)
        else:
            qH = pH
        H = qH.rsample()
        ## Conditional on the H, calculate  Z
        Z = self.z_pooler(X, H, G)

        ## Calculate predicted stamp
        pY = self.y_decoder(Z, X)

        return pH, qH, H, Z, pY

    def log_prob(self, X, S, Y):
        pH, qH, H, _, pY = self(X, S)
        log_pqh = pH.log_prob(H) - qH.log_prob(H)
        if self.fb_z is not None:
            log_pqh = log_pqh.clamp_max(self.fb_z)

        log_y = pY.log_prob(Y)
        elbo = log_pqh.sum() + log_y.sum()
        return elbo

    def forward(self, X, S):
        return self.encode(X, S)

    def predict(self, X, S, mean_Y=False, cond_output=False):
        n_inputs = S.size(0)
        pY = self.encode(X, S)[4]
        if mean_Y:
            Y = pY.loc.detach()
        else:
            Y = pY.sample()
        if cond_output:
            Y[:n_inputs] = S
        return Y


class KMeansDepGraph(nn.Module):
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def forward(self, X):
        km = KMeans(n_clusters=self.n_clusters)
        c = km.fit_predict(X.cpu().numpy())
        G = torch.zeros((len(c), self.n_clusters)).to(X.device)
        for i in range(self.n_clusters):
            G[c == i, i] = 1.0
        return G


# **************************
# Representation Poolers
# **************************
class AveragePooler(nn.Module):
    """
    Pools together representations by taking a sum.
    """

    def __init__(
        self,
        dim_z,
        f=None,
    ):
        super().__init__()
        self.dim_z = dim_z
        self.f = f

        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

    def forward(self, rep_R, GA):
        W = self.norm_graph(GA)
        if self.f:
            rep_R = self.f(rep_R)
        pz_all = torch.matmul(W, rep_R)
        return pz_all

class StarHNP(HNP):
    def __init__(self, stampsize, dz=4, fb_z=None, n_clusters=5):
        dy = stampsize ** 2
        dep_graph = KMeansDepGraph(n_clusters=n_clusters)
        z_inference = SequentialVarg(ConcatLayer([1]), MLP(dy, [16, 8], dz))
        dh = dz
        z_pooler = AttentionZPooler(2, dh, dz)

        h_prior = lambda X, G: Normal(
            torch.zeros(G.size(1), dz, device=G.device), torch.ones(G.size(1), dz, device=G.device)
        )

        # h_pooler = SimpleHPooler(dz)
        h_pooler = AttentionHPooler(2, dh, dz)

        y_decoder = SequentialVarg(
            ConcatLayer([0]),
            MLP(dz, [8, 16, 32], 2 * dy),
            SplitLayer(dy, -1),
            NormalEncoder(minscale=1e-7),
        )
        super().__init__(dep_graph, z_inference, z_pooler, h_prior, h_pooler, y_decoder, fb_z)


class SDSS_HNP(LightningModule):
    def __init__(
        self, stampsize=5, dz=4, sdss_dataset=None, max_cond_inputs=1000, n_clusters=5, fb_z=None
    ):
        super().__init__()
        self.sdss_dataset = sdss_dataset
        self.stampsize = stampsize
        self.stamper = StarStamper(self.stampsize)
        self.max_cond_inputs = max_cond_inputs
        self.n_clusters = n_clusters
        self.hnp = StarHNP(self.stampsize, dz, fb_z=fb_z, n_clusters=n_clusters)
        self.valid_losses = []

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        X, S, YY = self.prepare_batch(batch)
        loss = -self.hnp.log_prob(X, S, YY) / X.size(0)
        return loss

    def prepare_batch(self, batch, num_cond_inputs=None):
        X, img, locs = batch
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        if not isinstance(locs, torch.Tensor):
            locs = torch.from_numpy(locs)
        YY = self.stamper(img, locs[:, 1], locs[:, 0])[0]
        YY = YY.reshape(-1, self.stampsize ** 2)
        YY = (YY - YY.mean(1, keepdim=True)) / YY.std(1, keepdim=True)
        if num_cond_inputs is None:
            num_cond_inputs = self.max_cond_inputs
        S = YY[: min(YY.size(0), num_cond_inputs)]
        return X, S, YY

    def predict(self, X, S):
        out = self.hnp.predict(X, S, mean_Y=True)
        return out

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.valid_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(self.sdss_dataset, batch_size=None, batch_sampler=None, shuffle=True)


from sklearn.decomposition import PCA


class StarPCA:
    def __init__(self, k, n_clusters, stampsize) -> None:
        self.k = k
        self.n_clusters = n_clusters
        self.stampsize = stampsize
        self.stamper = StarStamper(self.stampsize)
        self.pca = PCA(n_components=self.k)
        self.mean = None
        self.dep_graph = KMeansDepGraph(n_clusters=self.n_clusters)

    def fit(self, Y):
        self.mean = Y.mean(0)
        self.pca.fit(Y.numpy())

    def predict(self, X, S, inverse_transform=True):
        n_inputs = S.size(0)
        # Get dependency graph
        G = self.dep_graph(X)

        # Transform S
        S_rep = torch.from_numpy(self.pca.transform(S)).float()

        # Average S based on cluster
        GT = G[:n_inputs].transpose(1, 0)
        GT = GT / GT.sum(dim=1, keepdim=True)
        cluster_rep = GT.matmul(S_rep)

        pred = G.matmul(cluster_rep)

        if inverse_transform:
            pred = self.pca.inverse_transform(pred)

        return pred

    def prepare_batch(self, batch, num_cond_inputs=None):
        X, img, locs = batch
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        if not isinstance(locs, torch.Tensor):
            locs = torch.from_numpy(locs)
        YY = self.stamper(img, locs[:, 1], locs[:, 0])[0]
        YY = YY.reshape(-1, self.stampsize ** 2)
        YY = (YY - YY.mean(1, keepdim=True)) / YY.std(1, keepdim=True)
        if num_cond_inputs is None:
            num_cond_inputs = YY.size(0)
        S = YY[: min(YY.size(0), num_cond_inputs)]
        return X, S, YY

    def fit_dataset(self, star_dataset):
        Ys = []
        for d in star_dataset:
            _, _, Y = self.prepare_batch(d)
            Ys.append(Y)
        Y = torch.cat(Ys, dim=0)
        self.fit(Y)


class SimpleZPooler(nn.Module):
    def __init__(self, dh):
        super().__init__()
        self.ap = AveragePooler(dh)

    def forward(self, X, H, G):
        return self.ap(H, G)


class SimpleHPooler(nn.Module):
    def __init__(self, dh):
        super().__init__()
        self.ap = AveragePooler(dh)

    def forward(self, X, Z, G):
        z_pooled = self.ap(Z, G)
        precis = 1.0 + G.sum(1)
        std = precis.reciprocal().sqrt().unsqueeze(1).repeat(1, z_pooled.size(1))
        return Normal(z_pooled, std)


from torch.nn import MultiheadAttention


class ResMLP(MLP):
    def __init__(self, in_features, hs, act=nn.ReLU, final=None):
        super().__init__(in_features, hs, in_features, act, final)

    def forward(self, X):
        r = super().forward(X)
        return X + r


class AttentionZPooler(nn.Module):
    def __init__(self, dim_x, dh, dz, num_heads_x=4, num_heads=2):
        super().__init__()
        self.dim_x = dim_x
        self.dim_x_embeded = dim_x * num_heads_x
        self.dh = dh
        self.dz = dz
        self.num_heads_x = num_heads_x
        self.num_heads = num_heads
        ##
        self.fc_x = ResMLP(self.dim_x_embeded, [8, 8, 8])
        self.fc_1 = ResMLP(self.dim_x_embeded * 2, [16, 16, 16])
        self.fc_2 = ResMLP(self.dim_x_embeded * 2, [16, 16, 16])
        self.linear = nn.Linear(self.dim_x_embeded * 2, self.dz)
        ## Averager for X
        self.ap = AveragePooler(dim_x)
        # First cross-attention block
        self.mab1 = MultiheadAttention(self.dim_x * self.num_heads_x, self.num_heads_x, vdim=dh)
        # First self-attention block
        self.sab1 = MultiheadAttention(self.dim_x_embeded * 2, self.num_heads)

    def forward(self, X, H, G):
        X_tilde = self.ap(X, G.t())

        X_embedded = torch.stack([X] * self.num_heads_x, dim=-1)
        X_embedded = X_embedded.reshape(X_embedded.size(0), 1, -1)
        X_embedded = self.fc_x(X_embedded)

        X_tilde_embedded = torch.stack([X_tilde] * self.num_heads_x, dim=-1)
        X_tilde_embedded = X_tilde_embedded.reshape(X_tilde_embedded.size(0), 1, -1)
        X_tilde_embedded = self.fc_x(X_tilde_embedded)

        Z = self.mab1(X_embedded, X_tilde_embedded, H.unsqueeze(1), need_weights=False)[0]
        Z = torch.cat([X_embedded, Z], dim=-1)
        Z = self.fc_1(Z)
        Z = self.sab1(Z, Z, Z)[0]
        Z = self.fc_2(Z)
        Z = self.linear(Z)
        return Z.squeeze(1)


class AttentionHPooler(nn.Module):
    def __init__(self, dim_x, dh, dz, num_heads_x=4, num_heads=2):
        super().__init__()
        self.dim_x = dim_x
        self.dim_x_embeded = dim_x * num_heads_x
        self.dh = dh
        self.dz = dz
        self.num_heads_x = num_heads_x
        self.num_heads = num_heads
        ##
        self.fc_x = ResMLP(self.dim_x_embeded, [8, 8, 8])
        self.fc_1 = ResMLP(self.dim_x_embeded * 2, [16, 16, 16])
        self.fc_2 = ResMLP(self.dim_x_embeded * 2, [16, 16, 16])
        self.linear = nn.Linear(self.dim_x_embeded * 2, 2 * self.dh)
        ## Averager for X
        self.ap = AveragePooler(dim_x)
        # First cross-attention block
        self.mab1 = MultiheadAttention(self.dim_x * self.num_heads_x, self.num_heads_x, vdim=dz)
        # First self-attention block
        self.sab1 = MultiheadAttention(self.dim_x_embeded * 2, self.num_heads)

    def forward(self, X, Zi, G):
        X_tilde = self.ap(X, G.t())

        Xi = X[: Zi.size(0)]

        X_embedded = torch.stack([Xi] * self.num_heads_x, dim=-1)
        X_embedded = X_embedded.reshape(X_embedded.size(0), 1, -1)
        X_embedded = self.fc_x(X_embedded)

        X_tilde_embedded = torch.stack([X_tilde] * self.num_heads_x, dim=-1)
        X_tilde_embedded = X_tilde_embedded.reshape(X_tilde_embedded.size(0), 1, -1)
        X_tilde_embedded = self.fc_x(X_tilde_embedded)

        H = self.mab1(X_tilde_embedded, X_embedded, Zi.unsqueeze(1), need_weights=False)[0]
        H = torch.cat([X_tilde_embedded, H], dim=-1)
        H = self.fc_1(H)
        H = self.sab1(H, H, H)[0]
        H = self.fc_2(H)
        H = self.linear(H)
        H = H.squeeze(1)
        mu, nu = torch.split(H, self.dh, -1)
        nu = 1e-7 + F.softplus(nu)
        return Normal(mu, nu)
