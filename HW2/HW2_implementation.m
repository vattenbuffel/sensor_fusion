function [X, P] = kalmanFilter(Y, x_0, P_0, A, Q, H, R)
%KALMANFILTER Filters measurements sequence Y using a Kalman filter. 
%
%Input:
%   Y           [m x N] Measurement sequence
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   A           [n x n] State transition matrix
%   Q           [n x n] Process noise covariance
%   H           [m x n] Measurement model matrix
%   R           [m x m] Measurement noise covariance
%
%Output:
%   x           [n x N] predicted state vector sequence
%   P           [n x n x N] Filter error convariance
%

% Parameters
N = size(Y,2);

n = length(x_0);
m = size(Y,1);

% Data allocation
x = zeros(n,N);
P = zeros(n,n,N);


% My code
%%%%%%%%%%%%%%%%%
% Estimate one step ahead in time.
[x, p] = linearPrediction(x_0, P_0, A, Q);
% Update the estimated position with the measurement
[x, p] = linearUpdate(x, p, Y(:, 1), H, R);      
X = x;
P(:,:,1) = p;
    
for i = 2:size(Y,2)
    % Estimate one step ahead in time.
    [x, p] = linearPrediction(X(:, end), P(:,:, i-1), A, Q);
    % Update the estimated position with the measurement
    [x, p] = linearUpdate(x, p, Y(:, i), H, R);           
    X = [X x];
    P(:,:,i) = p;
end
%%%%%%%%%%%%%%%%%
   
end

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
Vk = y - H*x; %Calculate the innovation
Sk = H*P*H.' + R; %Predict the covariance in yk
Kk = P*H.'*inv(Sk); %Calculate the Kalman gain, how much we trust the new measurement

x = x + Kk*Vk; %Estimate the new state
P = P - Kk*Sk*Kk.'; %Estimate the error covariance

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
X = mvnrnd(x_0, P_0).';


for i = 1:N
    % Every subsecent position is distributed normally with mean as A*x_previous and sigma Q
    X = [X mvnrnd(A*X(:,end), Q).'];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end