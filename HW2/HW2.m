clc; clear; close all;


tol = 1e-5;

% General parameters
n = 1;

% Measurement Parameters
H = 1;
R = .5^2;

% Prior
xPrior  = mvnrnd(zeros(n,1)', diag(ones(n)))';
V       = rand(n,n);
PPrior  = V*2*diag(rand(n,1))*V';

% Genereate measurement
y = mvnrnd(H*xPrior, H*PPrior*H' + R);

% Perform update
[xUpd, PUpd] = linearUpdate(xPrior, PPrior, y, H, R);
[xUpd_ref, PUpd_ref] = reference.linearUpdate(xPrior, PPrior, y, H, R);

% Plot results
figure(1); clf; hold on;
x = linspace(xUpd - 4*sqrt(PPrior), xUpd + 4*sqrt(PPrior),100);
plot(x, normpdf(x,xUpd, sqrt(PUpd)));
plot(x, normpdf(x,xPrior, sqrt(PPrior)));
plot(y,0,'sk', 'LineWidth', 2, 'MarkerSize', 10);
title('Your solution')
xlabel('x');
ylabel('p(x)')
legend('Updated density', 'Prior density', 'Measurement');

figure(2); clf; hold on;
x = linspace(xUpd_ref - 4*sqrt(PPrior), xUpd_ref + 4*sqrt(PPrior),100);
plot(x, normpdf(x, xUpd_ref, sqrt(PUpd_ref)));
plot(x, normpdf(x, xPrior, sqrt(PPrior)));
plot(y,0,'sk', 'LineWidth', 2, 'MarkerSize', 10);
title('Reference solution')
xlabel('x');
ylabel('p(x)')
legend('Updated density', 'Prior density', 'Measurement');


% Assert resutls
assert(isequal(size(xPrior),[n 1]), 'Dimension of prior and predicted mean need to be the same.');
assert(isequal(size(PPrior),[n n]), 'Dimension of prior and predicted covariance need to be the same.');
assert(all(abs(xUpd-xUpd_ref)<tol), 'Updated mean is not within tolerance.');
assert(all(all(abs(PUpd-PUpd_ref)<tol)), 'Updated covarinace is not within tolerance.');


[~, p] = chol(PUpd);
assert(p == 0 || trace((PUpd)) ~= 0, 'Posterior covariance is not positive semi definite covarinace matrix');




%%

function [x, P] = linearUpdate(x, P, y, H, R)
%LINEARPREDICTION calculates mean and covariance of predicted state
%   density using a linear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   y           [m x 1] Measurement
%   H           [m x n] Measurement model matrix
%   R           [m x m] Measurement noise covariance
%
%Output:
%   x           [n x 1] updated state mean
%   P           [n x n] updated state covariance
%

% Your code here
%%%%%%%%%%%%%%%%
Vk = y - H*x;
Sk = H*P*H.' + R;
Kk = P*H.'*inv(Sk);

x = x + Kk*Vk;
P = P - Kk*Sk*Kk.';

%%%%%%%%%%%%%%%%

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

% Your code here
%%%%%%%%%%%%%%%%
% The new gaussian distribution is just a linear transformation of the last so the only difference to mu is the matrix A, the noise mean is 0.
x = A*x;

% Same here. The new covariance is just the old but multiplied with mvnrnd "squared", and added noise Q variance
P = A*P*A.' + Q; 
%%%%%%%%%%%%%%%%

end

function Y = genLinearMeasurementSequence(X, H, R)
%GENLINEARMEASUREMENTSEQUENCE generates a sequence of observations of the state 
% sequence X using a linear measurement model. Measurement noise is assumed to be 
% zero mean and Gaussian.
%
%Input:
%   X           [n x N+1] State vector sequence. The k:th state vector is X(:,k+1)
%   H           [m x n] Measurement matrix
%   R           [m x m] Measurement noise covariance
%
%Output:
%   Y           [m x N] Measurement sequence
%

% your code here
%%%%%%%%%%%%%%%%
% Multiply all states with H and add the measurement noise.
Y = H*X + mvnrnd(zeros(size(R,1),1), R, size(X,2)).';

% Remove x0
Y = Y(:,2:end);
%%%%%%%%%%%%%%%%
end
function X = genLinearStateSequence(x_0, P_0, A, Q, N)
%GENLINEARSTATESEQUENCE generates an N-long sequence of states using a 
%    Gaussian prior and a linear Gaussian process model
%
%Input:
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   A           [n x n] State transition matrix
%   Q           [n x n] Process noise covariance
%   N           [1 x 1] Number of states to generate
%
%Output:
%   X           [n x N+1] State vector sequence
%

% Your code here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The first position is distributed normally with mean x_0 and sigma P_0
X = mvnrnd(x_0, P_0).'


for i = 1:N
    % Every subsecent position is distributed normally with mean as A*x_previous and sigma Q
    X = [X mvnrnd(A*X(:,end), Q).'];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end