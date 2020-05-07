clc; clear all; close all




% Test PF filter and compare with Kalman filter output. Generate 2D state sequence (CV, pos and vel) and 1D measurement sequence (pos) and compare
% outputs with Kalman filter.

% Set prior
sigma = 2;
x_0 = [0 1]';
P_0 = [sigma^2 0; 
       0 sigma^2];
n = size(x_0,1);

% Number of time steps
K = 20;

% Models
A = [1 0.1; 0 1];
Q = [0 0; 0 0.5];
H = [1 0];
R = 1;

m = 1;

% Generate state and measurement sequences
X = zeros(n,K);
Y = zeros(m,K);
   
q = mvnrnd([0 0], Q, K)';
r = mvnrnd(zeros(1,m), R, K)';
x_kmin1 = x_0;
for k = 1:K
    xk = f(x_kmin1,A) + q(:,k);
    X(:,k) = xk;
    x_kmin1 = xk;
    
    Y(:,k) = h(xk, H) + r(:,k);
end

% Run Kalman filter
[Xf, Pf] = kalmanFilter(Y, x_0, P_0, A, Q, H, R);

% Run PF filter with and without resampling
N = 20000;
proc_f = @(X_kmin1) (f(X_kmin1, A));
meas_h = @(X_k) (h(X_k, H));
plotFunc = @(k, Xk, Xkmin1, Wk, j) (0); % Define dummy function that does nothing

[xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, proc_f, Q, meas_h, R, N, false, plotFunc);
                          
[xfpr, Pfpr, Xpr, Wpr] = pfFilter(x_0, P_0, Y, proc_f, Q, meas_h, R, N, true, plotFunc);


% Compute means and covariances for each particle filter output

for k = 1:K
    mean_nRS = Xp(:,:,k) * Wp(:,k);
    cov_nRS = (Xp(:,:,k) - mean_nRS) * ((Xp(:,:,k) - mean_nRS)' .* Wp(:,k));
    % Compare with the provided means and covariances
    assert(norm(mean_nRS-xfp(:,k)) < 0.00001, 'The output mean from PF without RS is not consistent with the output particles.');
    assert(norm(cov_nRS-Pfp(:,:,k)) < 0.00001, 'The output covariance from PF without RS is not consistent with the output particles.');
    
    mean_RS = Xpr(:,:,k) * Wpr(:,k);
    cov_RS = (Xpr(:,:,k) - mean_RS) * ((Xpr(:,:,k) - mean_RS)' .* Wpr(:,k));
    % Compare with the provided means and covariances
    assert(norm(mean_RS-xfpr(:,k)) < 0.00001, 'The output mean from PF with RS is not consistent with the output particles.');
    assert(norm(cov_RS-Pfpr(:,:,k)) < 0.00001, 'The output covariance from PF with RS is not consistent with the output particles.');

    % Compare with Kalman filter
    assert(norm(Xf(:,k)-xfp(:,k)) < 0.5, 'The mean from PF WITHOUT resampling deviates too much from KF filter mean.');  
    assert(norm(Pf(:,:,k)-Pfp(:,:,k)) < 1, 'The covariance from PF WITHOUT resampling deviates too much from KF filter covariance.');  

    assert(norm(Xf(:,k)-xfpr(:,k)) < 0.5, 'The mean from PF WITH resampling deviates too much from KF filter mean.');  
    assert(norm(Pf(:,:,k)-Pfpr(:,:,k)) < 1, 'The covariance from PF WITH resampling deviates too much from KF filter covariance.'); 

end




function X_k = f(X_kmin1, A)
%
% X_kmin1:  [n x N] N states vectors at time k-1
% A:        [n x n] Matrix such that x_k = A*x_k-1 + q_k-1
    X_k = A*X_kmin1;
end

function H_k = h(X_k, H)
%
% X_k:  [n x N] N states
% H:    [m x n] Matrix such that y = H*x + r_k
    H_k = H*X_k;
end

function [X, P] = kalmanFilter(Y, x_0, P_0, A, Q, H, R)
%KALMANFILTER Filters measurements sequense Y using a Kalman filter. 
%
%Input:
%   Y           [m x N] Measurement sequence
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   A           [n x n] State transition matrix
%   Q           [n x n] Process noise covariance
%   H           [n x n] Measruement model matrix
%   R           [n x n] Measurement noise covariance
%
%Output:
%   x           [n x N] Estimated state vector sequence
%   P           [n x n x N] Filter error convariance
%

    % Parameters
    N = size(Y,2);

    n = length(x_0);
    m = size(Y,1);

    % Data allocation
    X = zeros(n,N);
    P = zeros(n,n,N);

    % Filter
    for k = 1:N

        if k == 1 % Initiate filter

            % Time prediction
            [xPred, PPred] = linearPrediction(x_0, P_0, A, Q);

        else

            % Time prediction
            [xPred, PPred] = linearPrediction(X(:,k-1), P(:,:,k-1), A, Q);

        end

        % Measurement update
        [X(:,k), P(:,:,k)] = linearUpdate(xPred, PPred, Y(:,k), H, R);

    end
end

function [x, P] = linearPrediction(x, P, A, Q)
%LINEARPREDICTION calculates mean and covariance of predicted state
%   density using a liear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   A           [n x n] State transition matrix
%   Q           [n x n] Process noise covariance
%
%Output:
%   x           [n x 1] predicted state mean
%   P           [n x n] predicted state covariance
%

% Predicted mean
x = A*x;

% Predicted Covariance 
P = A*P*A' + Q;

end

function [x, P] = linearUpdate(x, P, y, H, R)
%LINEARPREDICTION calculates mean and covariance of predicted state
%   density using a liear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   H           [n x n] Measruement model matrix
%   R           [n x n] Measurement noise covariance
%
%Output:
%   x           [n x 1] updated state mean
%   P           [n x n] updated state covariance
%

% Innovation
v = y - H*x;
S = H*P*H' + R;

% Kalman gain
K = P*H'/S;

% Updated mean and covariance
x = x + K*v;
P = P - K*S*K';

end


%
function [xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, proc_f, proc_Q, meas_h, meas_R, N, bResample, plotFunc)
%PFFILTER Filters measurements Y using the SIS or SIR algorithms and a
% state-space model.
%
% Input:
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   Y           [m x K] Measurement sequence to be filtered
%   proc_f      Handle for process function f(x_k-1)
%   proc_Q      [n x n] process noise covariance
%   meas_h      Handle for measurement model function h(x_k)
%   meas_R      [m x m] measurement noise covariance
%   N           Number of particles
%   bResample   boolean false - no resampling, true - resampling
%   plotFunc    Handle for plot function that is called when a filter
%               recursion has finished.
% Output:
%   xfp         [n x K] Posterior means of particle filter
%   Pfp         [n x n x K] Posterior error covariances of particle filter
%   Xp          [n x N x K] Particles for posterior state distribution in times 1:K
%   Wp          [N x K] Non-resampled weights for posterior state x in times 1:K

% Your code here, please. 
% If you want to be a bit fancy, then only store and output the particles if the function
% is called with more than 2 output arguments.


n = size(P_0,1);
K = size(Y,2);

xfp = [];
Pfp = zeros(n,n,K);
Xp = zeros(n,N,K);
Wp = [];

X = x_0 + mvnrnd(zeros(size(x_0)), P_0, N).';
W = mvnpdf(X.', x_0.', P_0.').';


for i = 1:K
    [X, W] = pfFilterStep(X, W, Y(:,i), proc_f, proc_Q, meas_h, meas_R);
    %plotFunc(X); %?????
    Wp = [Wp W.'];
    Xp(:,:,i) = X;
    xfp = [xfp X*W.'];
    Pfp(:,:,i) = (X - X*W.') * ((X - X*W.').' .* W.');
    
    if bResample 
        [X, W, j] = resampl(X, W);
    end
end


end

function [Xr, Wr, j] = resampl(X, W)
%RESAMPLE Resample particles and output new particles and weights.
% resampled particles. 
%
%   if old particle vector is x, new particles x_new is computed as x(:,j)
%
% Input:
%   X   [n x N] Particles, each column is a particle.
%   W   [1 x N] Weights, corresponding to the samples
%
% Output:
%   Xr  [n x N] Resampled particles, each corresponding to some particle 
%               from old weights.
%   Wr  [1 x N] New weights for the resampled particles.
%   j   [1 x N] vector of indices refering to vector of old particles

% Your code here!
% Normalise the weights and calculate the cumsum or the portions of 0->1 they own
W = W/(sum(W));
W = cumsum(W);

% Generate random places to take samples and sort them so that the looping is minimized
u = rand(size(W));
u = sort(u);

j = [];
Xr = [];

% Do the resampling
prev_index = -1;
next_index = 1;
for i = 1:size(u,2)
    while 0 < 1
        % If the first place is between 0 and the first weight
        if prev_index == -1 && u(i) < W(1)
            j = [1 j];
            Xr = [Xr X(:, 1)];
            break
            
        % If the place is inside the portion of 0->1 the weight own
        elseif prev_index ~= -1 && u(i) >= W(prev_index) && u(i) < W(next_index) 
            j = [next_index j];
            Xr = [X(:, next_index) Xr];
            break
        
        % The place was not in any place a weight owned, check the next weight
        else
            prev_index = next_index;
            next_index = next_index+1;
        end
            
    end
end

% All the samples are equally plausible
Wr = ones(1, size(W,2))/size(X,2);


end

function [X_k, W_k] = pfFilterStep(X_kmin1, W_kmin1, yk, proc_f, proc_Q, meas_h, meas_R)
%PFFILTERSTEP Compute one filter step of a SIS/SIR particle filter.
%
% Input:
%   X_kmin1     [n x N] Particles for state x in time k-1
%   W_kmin1     [1 x N] Weights for state x in time k-1
%   y_k         [m x 1] Measurement vector for time k
%   proc_f      Handle for process function f(x_k-1)
%   proc_Q      [n x n] process noise covariance
%   meas_h      Handle for measurement model function h(x_k)
%   meas_R      [m x m] measurement noise covariance
%
% Output:
%   X_k         [n x N] Particles for state x in time k
%   W_k         [1 x N] Weights for state x in time k

% Your code here!
% Sample q
X_k = proc_f(X_kmin1) + mvnrnd(zeros(size(X_kmin1, 1), 1) , (proc_Q), size(X_kmin1, 2)).';

% Calculate the weights
W_k = zeros(size(W_kmin1));
for i = 1:size(X_kmin1, 2)
   W_k(i) = W_kmin1(i) * normpdf(yk, meas_h(X_k(:, i)), sqrtm(meas_R));
end

% Normalize the weights
W_k = W_k / sum(W_k);
end
































