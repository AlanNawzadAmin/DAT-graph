import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from .parallel_linear_layer import ParallelNet


class DATTester:
    """ Step 2 of DAT-Graph -- learn the skeleton and psis from data.
    
    Parameters:
    moral: numpy array
        The inferred moral graph.
    device: device
    dtype: dtype
    hidden_width: int
    n_layers: int
    stats: function
        Takes in a batch of observations and returns batched
        statistics T_1(x), ..., T_L(x). T_1 must be the identity
        T_1(x)=x.

    Methods:
    get_importance(p=2)
        Return variable importances. This is our guess of the moral graph.
    train(train_dataloader, lr, n_steps, print_interval=100)
        Train the regressors.
    test(dataloader, n_steps)
        Get an estimate of the variance explained for each edge tested.
    get_psis()
        Get psis.
    get_test_edges_mbs()
        Returns an [n_test_edges, 2] matrix of the indices [i, j] of edges that were tested
        and a [n_test_edges, max_markov_blanket_size] matrix of the markov blanket of i for each
        tested edge [i, j].
    """
    def __init__(self, moral, device, dtype,
                 stats=lambda x:x[..., None],
                 hidden_width=100, n_layers=3):
        self.device = device
        self.dtype = dtype
        # save parameters
        self.n_nodes = len(moral)
        self.stats = stats
        # get number of stats
        dummy = torch.randn([1, self.n_nodes], dtype=self.dtype, device=self.device)
        self.n_stats = stats(dummy).shape[-1]
        assert torch.all(stats(dummy)[..., 0] == dummy), "T_1 must be the identity T_1(x)=x."
        # Configure the tests for each edge.
        self.__config_moral_graph(moral)
        
        # build nn
        self.n_layers = n_layers
        self.hidden_width = hidden_width
        self.__config_nn()

    def __config_moral_graph(self, moral):
        # Note the moral graph can contain some directed edges if we have identified
        # parents from intervention data.
        self.moral = moral
        symmetric_moral = (moral + moral.T) > 0
        # if we have an oriented edge, I'll assume we know it's adjacent
        uncertain_edges = moral * moral.T
        # get a vector of the edges to test
        self.test_edges = np.argwhere(uncertain_edges)
        self.n_test_edges = len(self.test_edges)
        
        # for each test edge i, j, get the indicies of the MB of i excluding j
        # and put these indicies into a matrix called moral_map.
        # we pad any extra entries with the value -1.
        moral_edges_per_node = [np.argwhere(m != 0)[:, 0] for m in symmetric_moral]
        # pad_to is the maximum number of variables to test ("M" in the manuscript)
        self.pad_to = symmetric_moral.sum(axis=-1).max()-1
        def pad_and_remove(array, remove, pad_to, pad_val=-1):
            r_array = array[np.logical_not(np.isin(array, [remove]))]
            return np.pad(r_array, (0,self.pad_to-len(r_array)), constant_values=pad_val)
        self.moral_map = [pad_and_remove(moral_edges_per_node[i], j, self.pad_to)
                          for i, j in self.test_edges]
        self.moral_map = np.array(self.moral_map)

        # Label sinks and parents in moral map
        self.sink_inds = np.argwhere(moral.sum(axis=0) == 0)[:, 0]
        self.sinks = torch.tensor(np.isin(self.moral_map, self.sink_inds))
        self.parents = [[(moral[i, l]!=0 and moral[l, i]==0) and (l != -1) for l in m]
                        for (i, j), m in zip(self.test_edges, self.moral_map)]
        self.parents = torch.tensor(np.array(self.parents))

        # make a data mapper that returns mb(i)-j, j, i
        def map_data(obs, intervention_mask):
            bs = len(obs)
            # we add an extra dimension of 0 so that -1 entries in moral map return 0
            data = torch.concat([obs, intervention_mask.to(self.device).to(self.dtype),
                                torch.zeros([bs, 1], device=self.device, dtype=self.dtype)], axis=-1)
            mb_x = data[:, self.moral_map]
            y = data[:, self.test_edges[:, 1]]
            x = data[:, self.test_edges[:, 0]]
            return mb_x, y, x
        self.map_data = map_data

    def __config_nn(self):
        # first we build the function that noises our observations x
        class ParameterizedAddNoise(nn.Module):
            def __init__(self, n_test_edges, pad_to, parents, sinks,
                         device, dtype):
                super().__init__()
                self.logits = torch.zeros([n_test_edges, pad_to], device=device, dtype=dtype)
                self.logits = torch.nn.Parameter(self.logits, requires_grad=True)
                self.eps = 1e-10
                with torch.no_grad():
                    self.logits[parents] = 1000
                    self.logits[sinks] = -1000

            def forward(self, x, hard=False):
                # make Laplace noise
                noise = torch.rand(*x.shape, device=x.device, dtype=x.dtype)
                lap_noise = torch.log(2 * torch.min(noise, 1-noise) + self.eps) * (2 * (noise < 0.5) - 1)
                # Scale it to give it thick tails
                abs_noise = torch.abs(lap_noise)
                lap_noise[abs_noise > 1] = (torch.sign(lap_noise[abs_noise > 1]) 
                                            * abs_noise[abs_noise>1] ** 1.1)
                noise = 0.5 * lap_noise
                ps = self.get_psis()
                x = x * ps + noise * (1 - ps)
                return x

            def get_psis(self):
                probs = torch.sigmoid(self.logits)
                return probs
            
        self.add_noise = ParameterizedAddNoise(self.n_test_edges, self.pad_to,
                                               self.parents, self.sinks,
                                               self.device, self.dtype)

        # Now set up neural networks
        self.regressor1 = ParallelNet(self.n_test_edges, self.pad_to, self.n_stats,
                                      self.hidden_width, self.n_layers)
        self.regressor2 = ParallelNet(self.n_test_edges, self.pad_to+1, self.n_stats,
                                      self.hidden_width, self.n_layers)
        self.regressor1 = torch.compile(self.regressor1.to(self.device).to(self.dtype))
        self.regressor2 = torch.compile(self.regressor2.to(self.device).to(self.dtype))

    def get_psis(self):
        return self.add_noise.get_psis().detach().cpu().numpy()

    def get_test_edges_mbs(self):
        return self.test_edges, self.moral_map

    def train(self, train_dataloader, lr, n_steps):
        losses_1 = np.empty([n_steps, self.n_test_edges])
        losses_2 = np.empty([n_steps, self.n_test_edges])
        optimizer_psi = torch.optim.Adam(self.add_noise.parameters(), lr=lr,
                                         betas=(0.9, 0.9), weight_decay=0)
        optimizer_nn = torch.optim.Adam(list(self.regressor1.parameters()) + list(self.regressor2.parameters()),
                                        lr=lr, weight_decay=0)
        def get_loss(dataloader, step):
            x, intervention_mask, _ = next(train_dataloader)
            x = x.to(self.device, non_blocking=True).to(self.dtype)
            mb_x, y, x = self.map_data(x, intervention_mask)
            
            # calculate loss L_1
            mb_x_noised = self.add_noise(mb_x)
            pred_y = self.regressor1(mb_x_noised)

            # assume T_1 is the identity can center y for other variables
            centred_y = y - pred_y[..., 0]
            residual = torch.concat([centred_y[..., None],
                                     (self.stats(centred_y.detach()) - pred_y)[..., 1:]],
                                    axis=-1)
            loss_1 = (residual**2).mean(axis=0).sum(axis=-1) # loss for each variable

            # calculate loss L_2
            x_mb_x = torch.concat([mb_x_noised, x[..., None]], axis=-1)
            pred_residual = self.regressor2(x_mb_x)
            if step % 2 != 0:
                regress_on = residual.detach()
            else:
                regress_on = residual
            error_2 = regress_on - pred_residual
            loss_2 = (error_2**2).mean(axis=0).sum(axis=-1)
            return loss_1, loss_2
            
        for step in (pbar := tqdm(range(n_steps))):
            loss_1, loss_2 = get_loss(train_dataloader, step)
            loss_1_loss_2 = loss_1.sum() + loss_2.sum()
            # backprop and update
            if step % 2 == 0:
                # update psis
                loss_3 = loss_1.sum() - loss_2.sum()
                optimizer_psi.zero_grad(set_to_none=True)
                loss_3.backward()
                optimizer_psi.step()
            else:
                # update regressors
                optimizer_nn.zero_grad(set_to_none=True)
                loss_1_loss_2.backward()
                optimizer_nn.step()
            pbar.set_description(f"Var. expl.: {loss_1.sum().detach().cpu().numpy().round(2)}"
                                 +f"Pred. error: {loss_1_loss_2.detach().cpu().numpy().round(2)}")
            # log edges
            losses_1[step] = loss_1.detach().cpu().numpy()
            losses_2[step] = loss_2.detach().cpu().numpy()
        
        # plot training curve
        plt.figure(figsize=[10, 5])
        plt.plot(losses_1.sum(-1), color='black', label='$L_1$')
        plt.plot(losses_2.sum(-1), color='red', label='$L_2$')
        plt.xlabel("step")
        plt.ylabel("error")
        plt.legend()
        plt.figure(figsize=[10, 5])
        ve = losses_1 - losses_2
        plt.plot(ve, color='blue', alpha=0.1)
        plt.plot(0*ve[:, 0], color='grey', alpha=0.8, ls='--')
        plt.xlabel("step")
        plt.ylabel("variance explained")
        plt.ylim(ve.min(-1)[-1] * 2, ve.max(-1)[-1] * 2)
        return losses_1, losses_2

    def test(self, dataloader, n_steps):
        total_loss_1 = 0.
        total_loss_2 = 0.
        with torch.no_grad():
            print("Estimating variance explained.")
            for step in tqdm(range(n_steps)):
                # format data
                x, intervention_mask, _ = next(dataloader)
                x = x.to(self.device, non_blocking=True).to(self.dtype)
                mb_x, y, x = self.map_data(x, intervention_mask)
            
                # accumulate loss L_1
                mb_x_noised = self.add_noise(mb_x)
                pred_y = self.regressor1(mb_x_noised)
                ## centred
                centred_y = y - pred_y[..., 0]
                residual = torch.concat([centred_y[..., None],
                                         (self.stats(centred_y.detach()) - pred_y)[..., 1:]],
                                        axis=-1)
                total_loss_1 = total_loss_1 + (residual**2).mean(axis=0).sum(axis=-1)

                # accumulate loss L_2
                x_mb_x = torch.concat([mb_x_noised, x[..., None]], axis=-1)
                pred_residual = self.regressor2(x_mb_x)
                total_loss_2 = total_loss_2 + ((residual - pred_residual) ** 2).mean(axis=0).sum(axis=-1)
        var_explained = (total_loss_1 - total_loss_2) / n_steps
        return var_explained.cpu().numpy()


