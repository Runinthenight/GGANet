# The Python implementation of GGANet.

GGANet is a geometry-enhanced gated attention network for drug-target interaction prediction

## Environment Setup
numpy==1.20.0  
paddlepaddle==2.2.2  
pandas==1.2.4  
rdkit==2021.3.1  
tqdm==4.67.1  
dgl==1.1.3  
dgllife==0.3.2  
fair-esm==2.0.0  
matplotlib==3.4.2  
networkx==3.1  
PyYAML==5.4  
scikit-learn==1.0.2  
torch==1.13.1  
pgl==2.1.5
networkx==3.1

## Instructions
1.Download pdb data from https://pan.baidu.com/s/1NRG2eOsNxEX5ayUQib9pBQ?pwd=ey6u.    
2.Unzip pdb_files.zip.  
2.Run process_data.py to process the data.  
3.Edit ./GGANet/config.py to modify the hyperparameters.  
4.Run run.py to train and test GGANet.  