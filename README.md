## Multiscale Recurrent Visual Attention


### Description of experiments
- **Baseline**: Standard/Original RAM model in (1)
- **Ratio-ed Glimpse**: The size of the original glimpse is ratio-ed to the size of the image. More specifically - max( [min(H,W) / (num_glimpses / 2)], [config_size] ). No scaling.
- **Reverse scaling, ratio-ed**: The size of the original glimpse is ratio-ed to the size of the image. More specifically - max( [min(H,W) / (num_glimpses / 2)], [config_size] ). Each glimpse is inverse scaled, so whereas standard RAM the glimpses get larger and more blurry, these get smaller and more fine-tuned. 


### Model Results

|       Model       |   Binary Acc.  |   # Best Epoch  | # Total Epochs  |  Improvement
|-------------------|----------------|-----------------|-----------------|--------------
| Baseline (OG RAM) |     65.571     |        54       |                 |       --
| Ratio-ed Glimpse  |     70.986     |        46       |        97       |     5.421%
|                   |                |                 |                 |



### References
1. https://arxiv.org/pdf/1406.6247.pdf
2. https://arxiv.org/pdf/2011.06190.pdf
3. Code: https://github.com/kevinzakka/recurrent-visual-attention


