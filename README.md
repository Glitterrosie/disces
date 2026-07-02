# DISCES
## Setup
1. Create the python environment: 
```bash
python3 -m venv venv && source venv/bin/activate
```
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. The code can then be run via the terminal:
```bash
python run_comparison.py --sample-size 5000 --trace-length 10 --dimensions 2 --type-count 5
```
- `--sample_size` = number of traces
- `--trace-length` = length of traces with `min=max`
- `--dimensions` = number of different attributes
- `--type-count` = number of possible values per attribute

## Algorithms
We distributed the basic DUC and DUS Algorithm in 3 different Versions:
### DUCT (DUC-Tree)
We distribute the search across this first level, assigning each branch to a separate worker to be explored independently, and combine the results at the end.

### DUCM (DUC-Matching)
We partition the traces across workers, so that each worker performs the match operation on its own subset of streams.

### DUSD (DUS-Dimension)
We distribute the per-attribute trees across workers so they are explored in parallel, then perform the merging sequentially once all trees are complete.