# FuzzyCliq
For CompBio

Assuming sample by gene data matrix `data` with `C` assumed sample clusters:

```python
from HybridSnnCluster import HybridSnnCluster

# create object of class HybridSnnCluster
hybrid_object = HybridSnnCluster(data, C)
# run the algorithm
hybrid_object.fit_model()
# access the assigned clusters
hybrid_object.clusters
```
