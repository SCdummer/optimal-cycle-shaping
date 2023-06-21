# Discovering Efficient Periodic Behaviours in Mechanical Systems via Neural Approximators

Code accompanying the paper:

**Discovering Efficient Periodic Behaviours in Mechanical Systems via Neural Approximators**\
Yannick Wotte, Sven Dummer, Nicolò Botteghi, Christoph Brune, Stefano Stramigioli, Federico Califano
The can be found at: [Link](https://arxiv.org/pdf/2212.14253.pdf).

![alt text](Figure_1.png)

**Abstract:** 
It is well known that conservative mechanical systems exhibit local oscillatory behaviours due to their elastic and gravitational potentials, which completely characterise these periodic motions together with the inertial properties of the system. 
The classification of these periodic behaviours and their geometric characterisation are in an ongoing secular debate, which recently led to the so-called eigenmanifold theory. 
The eigenmanifold characterises nonlinear oscillations as a generalisation of linear eigenspaces. With the motivation of performing periodic tasks efficiently, we use tools coming from this theory to construct an optimisation problem aimed at inducing desired closed-loop oscillations through a state feedback law. 
We solve the constructed optimisation problem via gradient-descent methods involving neural networks. Extensive simulations show the validity of the approach.

## Requirements

* Python 3.8
* Pytorch 
* pip install -r requirements.txt

## Learn optimal eigenmode

```bash
python learn_opt_eigenmode.py 
```

## Stabilize the system on the optimal oscillatory behaviour

```bash
python stabilize_opt_eigenmode.py 

```

## Cite
If you use this code in your own work, please cite our paper:
```
article{wotte2022discovering,
  title={Discovering Efficient Periodic Behaviours in Mechanical Systems via Neural Approximators},
  author={Wotte, Yannik and Dummer, Sven and Botteghi, Nicol{\`o} and Brune, Christoph and Stramigioli, Stefano and Califano, Federico},
  journal={arXiv preprint arXiv:2212.14253},
  year={2022}
}

```