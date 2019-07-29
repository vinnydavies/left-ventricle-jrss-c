# Fast Parameter Inference in a Biomechanical Model of the Left Ventricle using Statistical Emulation

Davies, V., Noè, U., Lazarus, A., Gao, H., Macdonald, B., Berry, C., Luo, X. and Husmeier, D. (2019). 
Fast Parameter Inference in a Biomechanical Model of the Left Ventricle using Statistical Emulation. 
*Accepted for publication, Journal of the Royal Statistical Society - Series C*.

## Instructions for use

There are two methods compared in the paper: the Local GP Method and the Low-Rank GP Method.
The code for these methods can be found in the folders called **method-localgp** (MATLAB code) and 
**method-low-rank-gps** (R code) respectively, with each folder containing a README file to explain 
how to use the code.

The simulated data is available at `\method-localgp\Simulations\Design4D`, while the data for the healthy
volunteer is available at `\method-localgp\Data\Stacked\DataHV.mat`.

## Further Details

Preprint: https://arxiv.org/abs/1905.06310