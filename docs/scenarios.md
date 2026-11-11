

**Q: I want to segment multiple classes (e.g. bone, muscle, organ) with different threshold range from the same data. How should I proceed?**

-   Run **seed generation separately** with different thresholds.
    


**Q: Should I use adaptive seed or sweep adaptive seed?**

-   Use **sweep adaptive seed** for many seeds, fast. Close thresholds give more seeds, also small ones from noise: raise `min_size` to drop them.
-   Use **adaptive seed** to reproduce results of the paper.
-   They give different seeds, so compare them on your own data. See [the comparison](./config_sweep_adaptive_seed.md#which-adaptive-seed-should-i-use).

**Q: Can I use a probability map of a model as the image?**

Yes. A float image takes decimal thresholds, for example `thresholds: [0.9, 0.7, 0.5]`, in the YAML and in napari.

**Q: Can I see intermediate results during SPROUT processing?**

Yes. Saving intermediate outputs can be useful if some results are **over-grown**, **over-split**, or if you want to understand how the SPROUT algorithm proceeds.

-   **Adaptive Seed Generation**  
    Set `save_every_iter` to True, store every iteration of seeds, and the intermediate adaptive seeds.
    
-   **Growth**  
    By default, results are saved at the end of each threshold range.  
    To save more frequently, adjust `save_every_n_iters`, which will save outputs every _n_ dilation steps.


**Q: How should I choose parameters for growth?**

As a rule of thumb:

-   Use **larger dilation steps in the early stages**. Early stopping prevents wasted computation, and this ensures that most components grow to similar sizes at each level.
    
-   Use **fewer dilation steps in later stages**, where the focus is on recovering fine details.
    
-   Increasing the **number of intermediate threshold ranges** provides finer control, making the growth process more stable and precise.


**Q: What if some regions are already at their boundaries, but others still need growth?**

Use the to_grow_ids parameter to grow only the selected regions.
This avoids leakage from completed regions while allowing unfinished ones to expand.
With napari-sprout, you can interactively choose which regions to grow for finer control.