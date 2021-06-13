# A causal protein-signaling networks derived from single-cell data.

## Description
- Statistics
  - Number of nodes: 11
  - Number of arcs: 17
  - Number of parameters: 178
  - Average Markov blanket size: 3.09
  - Average degree: 3.09
  - Maximum in-degree: 3
- [Graph](https://www.bnlearn.com/bnrepository/discrete-small.html#sachs)

## Download
- Download `data.raw.txt` (raw data) from https://pitt.box.com/s/kxd530qoxvwfciy9y63lrbp3q2p5b2ii and rename it to `data_concat.raw.txt`
- Download `sachs.bif.gz` (graph data) by running:
  ```bash
  $ wget http://www.bnlearn.com/bnrepository/sachs/sachs.bif.gz
  $ gunzip --keep sachs.bif.gz
  ```

## Reference
- Data set was downloaded via https://www.ccd.pitt.edu/wiki/index.php/Data_Repository .
- K. Sachs, O. Perez, D. Pe'er, D. A. Lauffenburger and G. P. Nolan. Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data. Science, 308:523-529, 2005.
