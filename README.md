## Multiscale Recurrent Visual Attention


### Description of experiments
- **Baseline**: Standard/Original RAM model in (1)
- **Ratio scaled to size**: The size of the original glimpse is ratio-ed to the size of the image. More specifically - max( [min(H,W) / (num_glimpses / 2)], [config_size] ). The glimpses are then scaled down to the the original glimpse size (50)
- **Ratio no scaling**: The size of the original glimpse is ratio-ed to the size of the image. More specifically - max( [min(H,W) / (num_glimpses / 2)], [config_size] ). The glimpses fed into there RATIOxRATIO size into the miniConv (i.e. no scaling whatsoever here)
- **Ratio interploate**: The size of the original glimpse is ratio-ed to the size of the image. More specifically: min(H,W) / 5. (5 is arbitrary here) Each glimpse is inverse scaled (interploated), so whereas standard RAM the glimpses get larger and more blurry, these get smaller and more fine-tuned. 2 scales with scale factor of .75
- **Continuous V1**: inputs flattened h_t (from each scale image) into a fully connected lienar layer for a final contunous prediction
- **Continuous V2** Inputs fully connected layer into a final fully connected layer kiond of like the Stacked Generalization model with the schoolsz


### Classification Model Results

|       Model            |     Acc.       |    r2   |   # Best Epoch  | # Total Epochs  |  Improvement
|------------------------|----------------|---------|-----------------|-----------------|---------------
| Baseline (OG RAM)      |     66.73%     |         |        83       |       132       |      --
| Ratio scaled to size   |     70.99%     |         |        46       |        97       |     4.26%
| Ratio no scaling       |     67.89%     |         |        91       |       142       |     1.16%
| Ratio interpolate      |     70.60%     |         |       114       |       155       |     3.87%
| Continuous (imagery)   |    1280.28     | .55/.6  |       161       |       200       |      --
| Continuous (i + c) v1  |    1387.54     |   .48   |       129       |       180       |  * Census data in h_t before all of the FC layers
| Continuous (i + c) v2  |    1391.49     |         |        48       |                 |  * Census data placed only in fc_cont 


### References
1. https://arxiv.org/pdf/1406.6247.pdf
2. https://arxiv.org/pdf/2011.06190.pdf
3. Code: https://github.com/kevinzakka/recurrent-visual-attention
4. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.97.9314&rep=rep1&type=pdf
5. https://github.com/cayetanobv/raster2xyz
6. https://papers.nips.cc/paper/2014/file/09c6c3783b4a70054da74f2538ed47c6-Paper.pdf
