# VLAD: Task-agnostic VAE-based lifelong anomaly detection

> **NOTE**: This is a snapshot of code from a larger project that evolved since then.
> At this moment, the code is not guaranteed to work out of the box and may be a little messy. We plan to include VLAD in
> the [pyCLAD](https://pypi.org/project/pyclad/) library in the future.


# Introduction
VLAD's main goal is to provide a task-agnostic anomaly detection algorithm that can be used in a lifelong learning setting.

Main class containing `VLAD` algorithm is available in `our.py` module. In this repository, `VLAD` is named as `OurModel`.


## Paper & Citation

The details of the algorithm are described in the paper published in Neural Networks journal - 
[see here](https://www.sciencedirect.com/science/article/abs/pii/S0893608023002733). 
Full text is also available
on [ResearchGate](https://www.researchgate.net/publication/375957959_VLAD_Task-Agnostic_VAE-based_Lifelong_Anomaly_Detection)

Whenever using the algorithm or generated scenarios, please cite:

K. Faber, R. Corizzo, B. Sniezynski and N. Japkowicz, "Faber, Kamil, et al. "Vlad: Task-agnostic vae-based lifelong anomaly detection," in Neural Networks, vol. 165, pp. 248-273, 2023, doi:
10.1016/j.neunet.2023.05.032

```
@article{faber2023vlad,
  title={Vlad: Task-agnostic vae-based lifelong anomaly detection},
  author={Faber, Kamil and Corizzo, Roberto and Sniezynski, Bartlomiej and Japkowicz, Nathalie},
  journal={Neural Networks},
  volume={165},
  pages={248--273},
  year={2023},
  publisher={Elsevier}
}

```