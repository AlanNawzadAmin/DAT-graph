# Causal discovery with a Differentiable Adjacency Test

<p align="center">
<img width="500" alt="image" src="https://github.com/AlanNawzadAmin/DAT-graph/assets/70599003/eefaee93-4d36-47cd-ae49-245a70225083">
</p>

The repo includes code implementing DAT and DAT-Graph from [Scalable and Flexible Causal Discovery with an Efficient Test for Adjacency](https://arxiv.org/abs/2406.09177) appearing in ICML 2024.
Authors are [Alan Nawzad Amin](https://alannawzadamin.github.io/) and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

The notebook `run_DAT_graph.ipynb` includes code to generate synthetic data and analyze it using DAT-Graph.
To run the notebook you will need a GPU and you will need to install dependencies.
To do so, you can run the following code
```
conda create --name dat_graph python==3.10 -y
conda activate dat_graph
conda install pip -y
pip install .
python -m ipykernel install --user --name dat_graph
```
