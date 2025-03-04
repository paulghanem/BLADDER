# Biomechanistic Learning Augmentation of Deep Differential Equation Representations (BLADDER)
Learning hidden dynamics of the Lower Urinary Tract using a combination of differential equations and neural networks, where neural afferent signal dynamics of the animal are unknown and replaced by a neural network to be learned without neural afferent signal data available. 





## Usage/Examples
-To run the simulation that learns Nba without measurement (just from volume and pressure)
```bash
$ python dual_EKF_noNba_1.4.py 
```
## Experiment run
- extract volume and pressure data from any dataset and store it it V.mat and P.mat
```bash
$ python dual_EKF_1.4_experiment.py 
```



## Installation
-create conda virtual environment using python 3.8 
```bash
$ pip install requirements.txt

```


