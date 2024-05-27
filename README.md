# DAT-Graph

The repo includes code implementing DAT and DAT graph from "Scalable and Flexible Causal Discovery with an Efficient Test for Adjacency".
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