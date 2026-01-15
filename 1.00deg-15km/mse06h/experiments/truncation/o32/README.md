# Create the mapping to and from o32 grid

First I had to find the cutoff radius that resulted in no isolated nodes.
I did that by creating the graph and looking at the inspection

```
anemoi-graphs create recipe.truncation.yaml graph.pt > inspection.txt
anemoi-graphs inspect graph.pt figures
```

Then to create the weights

```
anemoi-graphs export_to_sparse recipe.truncation.yaml truncation-matrices/ --edge-attribute-name gauss_weight
```

