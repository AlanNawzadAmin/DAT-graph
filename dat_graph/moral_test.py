import numpy as np
from matplotlib import pyplot as plt
import torch
from .parallel_linear_layer import ParallelNet

class MoralTester():
    """ Step 1 of DAT-Graph -- learn the moral graph from data.
    
    Parameters:
    n_nodes: int
    device: device
    dtype: dtype
    intervention_inds: list
    stats: function
        Takes in a batch of observations and returns batched
        statistics T_1(x), ..., T_L(x). T_1 must be the identity
        T_1(x)=x.
    hidden_width: int
    n_layers: int

    Methods:
    get_importance(p=2)
        Return variable importances. This is our guess of the moral graph.
    train(train_dataloader, lr, n_steps, sparsity, print_interval=100)
        Train the regressor.
    """
    def __init__(self, n_nodes, device, dtype,
                 intervention_inds=[], stats=lambda x:x[..., None],
                 hidden_width=100, n_layers=3):
        self.device = device
        self.dtype = dtype
        # save parameters
        self.n_nodes = n_nodes + len(intervention_inds)
        self.intervention_inds = intervention_inds
        self.stats = stats
        # get number of stats
        dummy = torch.randn([1, self.n_nodes], dtype=self.dtype, device=self.device)
        self.n_stats = stats(dummy).shape[-1]
        assert torch.all(stats(dummy)[..., 0] == dummy), "T_1 must be the identity T_1(x)=x."

        # make neural network and mask
        self.mask = (1 - torch.eye(self.n_nodes, dtype=self.dtype, device=self.device))
        self.regressor = ParallelNet(self.n_nodes, self.n_nodes,self.n_stats,
                                     hidden_width, n_layers)
        self.regressor = torch.compile(self.regressor.to(self.device).to(self.dtype))

    def get_importance(self, p=2):
        sparse = self.regressor.l1_mat(p=p)
        return sparse

    def train(self, train_dataloader, lr, n_steps,
              sparsity, print_interval=1000,):
        losses = []
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr)
        def get_loss(dataloader):
            x, intervention_mask, _ = next(dataloader)
            if len(self.intervention_inds) > 0:
                intervention_indicators = 1-intervention_mask[:, self.intervention_inds].to(self.dtype)
                x = torch.concat([x, intervention_indicators], axis=-1)
            x = x.to(self.device).to(self.dtype)
            x_m = torch.einsum('...i, ij->...ji', x, self.mask)
            
            preds = self.regressor(x_m)
            # assume first statistic is T_1(x)=x
            pred_exp = preds[..., 0]
            loss = (x - pred_exp) ** 2
            if self.n_stats > 0:
                # Predict the rest of the statistics for the centred variable T_l(x - E[x]).
                # Thus we predict Var(x) instead of Ex^2.
                centered_stats = self.stats(x - pred_exp.detach())[..., 1:]
                loss += ((centered_stats - preds[..., 1:]) ** 2).sum(axis=-1)
            mean_err = loss.mean(axis=0)
            ## sparsity
            l1 = self.regressor.l1_mat().sum(axis=-1)
            return mean_err, sparsity * mean_err.detach() * l1

        for step in range(n_steps):
            mean_err, l1 = get_loss(train_dataloader)
            loss = (mean_err + l1).sum()
            # backprop and update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (step-1) % print_interval == 0:
                print("step:", step, '/', n_steps, ", mean error:", mean_err.sum().detach().cpu().numpy())
            losses.append(loss.sum().detach().cpu())

        # plot loss curve
        plt.figure(figsize=[5, 5])
        plt.plot(losses, color='black')
        plt.ylabel("loss")
        plt.xlabel("step")
