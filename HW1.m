%% 1 a
clc; clear; close all;


A = [1 1; 1 -1];
b = [0;0];
f = @(x) A*x + b;
mu = [10;0];
sigma = [0.2 0; 0 8];

[calc_mu, calc_sigma] = affineGaussianTransform(mu, sigma, A, b);
calc_level_curve = sigmaEllipse2D(calc_mu, calc_sigma, 3, 256);

[app_mu, app_sigma, app_ys] = approxGaussianTransform(mu, sigma, f);
app_level_curve = sigmaEllipse2D(app_mu, app_sigma, 3, 256);


plot(app_ys(1, :), app_ys(2, :))
hold on
plot(calc_level_curve(1,:), calc_level_curve(2,:))
plot(app_level_curve(1,:), app_level_curve(2,:))



%Vad menar dom plotta mean?
% Both the plots fit well, they fit better the higher N is because of the
% law of big numbers.



%% 1 b
clc; clear; close all;

N = 10000000;

f = @(x) [vecnorm(x); atan2(x(2, :), x(1, :)) ];
mu = [10;0];
sigma = [0.2 0; 0 8];

[app_mu, app_sigma, app_ys] = approxGaussianTransform(mu, sigma, f, N);
app_level_curve = sigmaEllipse2D(app_mu, app_sigma, 3, 256);


plot(app_ys(1, :), app_ys(2, :))
hold on
plot(app_level_curve(1,:), app_level_curve(2,:))


%% 2 a
clc; clear; close all;

mu_haf = 1.1; 
sigma_haf = 0.5^2;
sigma_Anna = 0.2^2;

mu_kvit = 1; 
sigma_kvit = 0.5^2;
sigma_Else = 1^2;

[mu_haf, sigma_haf] = jointGaussian(mu_haf, sigma_haf, sigma_Anna);
haf_level_curve = sigmaEllipse2D(mu_haf, sigma_haf, 3, 256);

[mu_kvit, sigma_kvit] = jointGaussian(mu_kvit, sigma_kvit, sigma_Else);
kvit_level_curve = sigmaEllipse2D(mu_kvit, sigma_kvit, 3, 256);


plot(kvit_level_curve(1,:), kvit_level_curve(2,:))
hold on
plot(haf_level_curve(1,:), haf_level_curve(2,:))

legend('Haf', 'Kvit')


%Dependency is bigger in kvit





%%
function [ xy ] = sigmaEllipse2D( mu, Sigma, level, npoints )
%SIGMAELLIPSE2D generates x,y-points which lie on the ellipse describing
% a sigma level in the Gaussian density defined by mean and covariance.
%
%Input:
%   MU          [2 x 1] Mean of the Gaussian density
%   SIGMA       [2 x 2] Covariance matrix of the Gaussian density
%   LEVEL       Which sigma level curve to plot. Can take any positive value, 
%               but common choices are 1, 2 or 3. Default = 3.
%   NPOINTS     Number of points on the ellipse to generate. Default = 32.
%
%Output:
%   XY          [2 x npoints] matrix. First row holds x-coordinates, second
%               row holds the y-coordinates. First and last columns should 
%               be the same point, to create a closed curve.


%Setting default values, in case only mu and Sigma are specified.
if nargin < 3
    level = 3;
end
if nargin < 4
    npoints = 32;
end

% Create a vector of angles. The angles are those to create the level curve of the distribution.
% The vector starts at 0 and ends at 0 in order to creat a full elipse. In between the zeros there are npoints-2 points.
fi = [0:2*pi/(npoints-1):2*pi-2*pi/(npoints-1) 0];

xy = level*sqrtm(Sigma)*[cos(fi); sin(fi)] + mu;

end

function [mu, Sigma] = jointGaussian(mu_x, sigma2_x, sigma2_r)
%jointGaussian calculates the joint Gaussian density as defined
%in problem 1.3a. 
%
%Input
%   MU_X        Expected value of x
%   SIGMA2_X    Covariance of x
%   SIGMA2_R    Covariance of the noise r
%
%Output
%   MU          Mean of joint density 
%   SIGMA       Covariance of joint density


%Your code here

% mu makes intuitive sense. The first variable is just x so it's mu is mu_x. The second variable is x+r and since it's mu is mu_x+mu_r, mu_r is 0 so it's mu is also mu_x.
mu = [mu_x; mu_x];

% Element 1,1 in the cov matrix is simply sigma2_x.
% Element 1,2 in the cov matrix is cov(x,y) = E[xy] - E[x]E[y] = E[x^2 + xr] - 0. Since they're uncorrelated E[x^2 + xr] = E[x^2] = sigma2_x
% Element 2,1 in the cov matrix is cov(x,y) = E[xy] - E[x]E[y] = E[x^2 + xr] - 0. Since they're uncorrelated E[x^2 + xr] = E[x^2] = sigma2_x
% Element 2,2 in the cov matrix is simply sigma2_x.
Sigma = [sigma2_x sigma2_x; sigma2_x sigma2_r+sigma2_x];

end

function [mu_y, Sigma_y] = affineGaussianTransform(mu_x, Sigma_x, A, b)
%affineTransformGauss calculates the mean and covariance of y, the 
%transformed variable, exactly when the function, f, is defined as 
%y = f(x) = Ax + b, where A is a matrix, b is a vector of the same 
%dimensions as y, and x is a Gaussian random variable.
%
%Input
%   MU_X        [n x 1] Expected value of x.
%   SIGMA_X     [n x n] Covariance of x.
%   A           [m x n] Linear transform matrix.
%   B           [m x 1] Constant part of the affine transformation.
%
%Output
%   MU_Y        [m x 1] Expected value of y.
%   SIGMA_Y     [m x m] Covariance of y.

%Your code here

% This comes from the definitions. 
% E[Y] = E[L*X+b] = L*E[X]+b
% E[(Y-mu_y)(Y-mu_y)^T] = E[(L*X+b-E[Y])(L*X+b-E[Y])^T] = A*Sigma_x*A^T

mu_y = A*mu_x + b;
Sigma_y =  A*Sigma_x*A.';
    
end


function [mu_y, Sigma_y, y_s] = approxGaussianTransform(mu_x, Sigma_x, f, N)
%approxGaussianTransform takes a Gaussian density and a transformation 
%function and calculates the mean and covariance of the transformed density.
%
%Inputs
%   MU_X        [m x 1] Expected value of x.
%   SIGMA_X     [m x m] Covariance of x.
%   F           [Function handle] Function which maps a [m x 1] dimensional
%               vector into another vector of size [n x 1].
%   N           Number of samples to draw. Default = 5000.
%
%Output
%   MU_Y        [n x 1] Approximated mean of y.
%   SIGMA_Y     [n x n] Approximated covariance of y.
%   ys          [n x N] Samples propagated through f


if nargin < 4
    N = 5000;
end

%Your code here
% Create N random samples
x_s = mvnrnd(mu_x, Sigma_x, N).';
% Take these samples through the function
y_s = f(x_s);

% Calculate the sample mean
mu_y = (sum(y_s.').')/N;
% Calculate the sample covariance
Sigma_y = ((y_s-mu_y)*(y_s-mu_y).')/(N-1);

end
