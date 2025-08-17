**Q: How should I choose parameters for growth?**

As a rule of thumb:

-   Use **larger dilation steps in the early stages**. Early stopping prevents wasted computation, and this ensures that most components grow to similar sizes at each level.
    
-   Use **fewer dilation steps in later stages**, where the focus is on recovering fine details.
    
-   Increasing the **number of intermediate threshold ranges** provides finer control, making the growth process more stable and precise.