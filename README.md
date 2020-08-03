### infomaxANE (Attributed network embedding based on mutual information estimation)

#### Requierments

* PyTorch 1.3.1
* Python 3.6

#### Usage

* **Transductive learning**

Run `python main.py` in infomaxANE-transductive, the parameter description is written in main.py

For example,

```shell
python main.py --cuda True --dataset cora --clips 4 --alpha 2 --beta 0.5
```



* **Inductive learning**

--STEP 1

Run `sh reddit.sh` in infomaxANE-inductive to run infomaxANE for dataset reddit;

Run `sh ppi.sh` in infomaxANE-inductive to run infomaxANE for dataset PPI;

--STEP 2

Run `python eval_reddit.py` to evaluate learned embedding with the same evaluation method in DGI/GMI;

Run`python eval_ppi.py` to evaluate learned embedding with the same evaluation method in DGI/GMI

* **Data**
Cora, Citeseer, Wiki, Pubmed are available in https://github.com/gaoghc/DANE ;
PPI, Reddit are available in http://snap.stanford.edu/graphsage/

*acknowledgements*

Our codes are adapted from public sources in https://github.com/williamleif/graphsage-simple and https://github.com/williamleif/GraphSAGE, thanks them for sharing.

