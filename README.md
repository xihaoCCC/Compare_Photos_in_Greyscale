# Comparasion two images' Greyscales by Statistical tools 

## Summary
In this project, I analyzed two photos taken at the Suzzallo library at the University of Washington: one with me present and one without. 
To assess their similarity, I utilized statistical measures including Kullback–Leibler divergence, Jensen–Shannon divergence, 
and the Kolmogorov–Smirnov test (KS test). These methods compare the greyscale distributions of the images. The primary difference in the
photos is my presence, occupying roughly 10% of the frame. The results consistently indicate a high similarity between the two images.

## Setup
This project is developed using Python 3. Refer to `main.ipynb` for the required packages.


## File Descriptions
- `main.ipynb`: Python notebook responsible for data generation and model construction.
- `fit_gaussian.html`: HTML export of the `main.ipynb` notebook.
- `presentation.mp4`: A brief video presentation discussing the work and key concepts of the project.


## Content
This project includes 7 parts in total, specifically: 
1. Synthesize a multimodal Gaussian distribution
2. Fit a piecewise linear regression model
3. Fit three spline models with 2, 3, and 4 knots respectively
4. Compare the R-squared values and root mean square deviations (RMSD) of the models from the previous sections
5. Fit four polynomial models with degree 2,3,4,5 respectively
6. Compare the fitting times of the constructed models
7. Construct two polynomial models of degree 5 using Lasso and Ridge regularization techniques

### For inquiries or further discussion, please reach out to me at [xihaocao@163.com](mailto:xihaocao@163.com). Thank you!
