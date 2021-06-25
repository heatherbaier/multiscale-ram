## Multiscale Recurrent Visual Attention


### Description of experiments
- **Baseline**: Standard/Original RAM model in (1)
- **Ratio scaled to size**: The size of the original glimpse is ratio-ed to the size of the image. More specifically - max( [min(H,W) / (num_glimpses / 2)], [config_size] ). The glimpses are then scaled down to the the original glimpse size (50)
- **Ratio no scaling**: The size of the original glimpse is ratio-ed to the size of the image. More specifically - max( [min(H,W) / (num_glimpses / 2)], [config_size] ). The glimpses fed into there RATIOxRATIO size into the miniConv (i.e. no scaling whatsoever here)
- **Ratio interploate**: The size of the original glimpse is ratio-ed to the size of the image. More specifically: min(H,W) / 5. (5 is arbitrary here) Each glimpse is inverse scaled (interploated), so whereas standard RAM the glimpses get larger and more blurry, these get smaller and more fine-tuned. 


### Model Results

|       Model           |   Binary Acc.  |   # Best Epoch  | # Total Epochs  |  Improvement
|-----------------------|----------------|-----------------|-----------------|--------------
| Baseline (OG RAM)     |     66.73%     |        83       |       132       |       --
| Ratio scaled to size  |     70.99%     |        46       |        97       |     3.27%
| Ratio no scaling      |     67.12%     |        36       |                 |    0.195%
| Ratio interpolate     |                |                 |                 |


### References
1. https://arxiv.org/pdf/1406.6247.pdf
2. https://arxiv.org/pdf/2011.06190.pdf
3. Code: https://github.com/kevinzakka/recurrent-visual-attention


