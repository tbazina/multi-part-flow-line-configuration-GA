# Optimisation for multi-part flow-line configuration of reconfigurable manufacturing system using GA

Optimization model was developed by [(Dou, Dai & Meng, 2010)](#references).
Afore-mentioned model was implemented in Python.

## Getting started

Input Data is defined in script

```
main.py
```

Script is programmed for running inside [Jupyter Notebook](https://jupyter.org/):

```
Main.ipynb
```

Example output data is given in:

```
Main.html
```

### Prerequisites

Scrpts were made using:
* [Pygmo](https://esa.github.io/pagmo2/)
* [NumPy](https://github.com/numpy/numpy)
* [numba](https://github.com/numba/numba)
* [pandas](https://github.com/pandas-dev/pandas)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [NetworkX](https://github.com/networkx)

### Content

Model components are in their corresponding scripts:
```
problem -> problem.py
algorithm -> algorithm.py
population -> population.py
```

## References

* Dou, J., Dai, X., & Meng, Z. (2010). Optimisation for multi-part flow-line configuration of reconfigurable manufacturing system using GA. International Journal of Production Research, 48(14), 4071-4100.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
