# Market-and-Strategic-Interaction-in-Network
By Leo Pan 030025552 and Nicolas Piker 029966545
## Dependencies
We installed the Python `networkx` and `matplotlib` packages. 
## Running the program
When creating the `gml` file, please make sure the seller nodes have the attributes `id`, `label`, and `price`. The buyer nodes need `id` and `label`. The edges from the buyer to the seller need to have an attribute `valuation`. The nodes must match their `id` and `label`, and they must start at 0. The nodes must also be labeled without skipping any numbers. 

The `--plot` flag will display an image of the nodes with all edges visible.

The `--interactive` flag will display a graph with only the preferred paths, and it will show every round. 
