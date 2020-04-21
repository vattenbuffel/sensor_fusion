%% 1
clc; clear; close all;

Q = 1.5;
R = 2.5;
x0 = 10;
P0 = 2;
N = 20;

x = genLinearStateSequence(x0, P0, 1, Q, N);
y = genLinearMeasurementSequence(x, 1, R);

%a
plot(x)
hold on
plot(y)
legend('x', 'y')
title('a')

%b
[filtered_y, cov] = kalmanFilter(y, x0, P0, 1, Q, 1, R);
figure
plot(x(2:end))
hold on
plot(y)
plot(filtered_y)
plot([filtered_y.'] + 3*sqrt([cov(:)]), '--b')
plot([filtered_y.'] - 3*sqrt([cov(:)]), '--b')
legend('x', 'y', 'y_{filtered}', '3 \sigma', '-3 \sigma')
title('b')


figure
xs = 2:0.1:20;
ys = normpdf(xs, filtered_y(1), sqrt(cov(:,:,1)));
plot(xs, ys)
hold on
plot(x(2), 0, 'o')
title('b')
legend('P(x)', 'x_{true}')

figure
ys = normpdf(xs, filtered_y(end/2), sqrt(cov(:,:,end/2)));
plot(xs, ys)
hold on
plot(x((end+1)/2), 0, 'o')
title('b')
legend('P(x)', 'x_{true}')

figure
ys = normpdf(xs, filtered_y(end), sqrt(cov(:,:,end)));
plot(xs, ys)
hold on
plot(x(end-1), 0, 'o')
title('b')
legend('P(x)', 'x_{true}')

%c
close all
figure
ys = normpdf(xs, filtered_y(10), sqrt(cov(:,:,10)));
plot(xs, ys)
title('c')

hold on
[new_x, new_cov] =  linearPrediction(filtered_y(10), cov(:,:,10), 1, Q);
ys = normpdf(xs, new_x, new_cov); 
plot(xs, ys)
title('c')

ys = normpdf(xs, filtered_y(11), sqrt(cov(:,:,11)));
plot(xs, ys)
xline(y(11), 'b');
xline(x(12), 'black');
legend('p(x_{k-1}|y_{1:k-1})', 'p(x_{k}|y_{1:k-1})', 'p(x_{k}|y_{1:k})', 'y(k)', 'x(k)') %Filtered, predicted, filtered
title('c')
%% 1d
clc; clear; close all;

Q = 1.5;
R = 2.5;
x0 = 2;
P0 = 6;
N = 1000;

x = genLinearStateSequence(x0, P0, 1, Q, N);
y = genLinearMeasurementSequence(x, 1, R);
[filtered_y, cov, Vk] = kalmanFilter(y, x0, P0, 1, Q, 1, R);
x = x(2:end);

estimated_mean = sum(filtered_y)/size(filtered_y,2);
estimation_error = x-filtered_y;

histo = histogram(estimation_error, 'Normalization', 'pdf'); % Normal distributed
hold on
posterior = normpdf(-10:10, 0, sqrt(cov(end)));
plot(-10:10, posterior)
legend('histogram', 'P(0, \surd{P_{K|K})}')

figure
Vk_mean = sum(Vk)/size(Vk, 2);
autocorr(Vk)



%% 1f
clc; clear; close all;
Q = 1.5;
R = 2.5;
x0 = 2;
P0 = 6;
N = 10;

x = genLinearStateSequence(x0, 2.6, 1, Q, N);
y = genLinearMeasurementSequence(x, 1, R);
[filtered_y, cov, Vk] = kalmanFilter(y, x0, P0, 1, Q, 1, R);
x = x(2:end);

[filtered_y_wrong_prior, cov, Vk] = kalmanFilter(y, 10, P0, 1, Q, 1, R);

plot(x)
hold on
plot(filtered_y)
plot(filtered_y_wrong_prior)
legend('x', 'filtered y', 'filtered y with wrong x_0')



%% 2
clc; clear; close all;
Q = [0 0; 0 1.5];
R = 2;
x0 = [1; 3]; 
P0 = 4*eye(2);
N = 100;
T = 0.01;

A = [1 T; 0 1];
H = [1 0];

x = genLinearStateSequence(x0, P0, A, Q, N);
y = genLinearMeasurementSequence(x, H, R);
[filtered_y, cov, Vk] = kalmanFilter(y, x0, P0, A, Q, H, R);
x = x(:, 2:end);

%a
%plot pos
plot(x(1,:))
hold on
plot(y(1,:))
legend('Predicted pos', 'measured pos')
title('a')
% very low motion noise but failry high measurement noise so it makes sense

%plot vel
figure
plot(x(2,:))
legend('Predicted vel')
title('a')
% makes sense, it's pretty much constant but with a small addative motion
% noise


%b
%plot pos
figure
plot(x(1,:))
hold on
plot(y)
plot(filtered_y(1,:))
title('b')

fitt_matlab = cov(1,1,:);
fitt_matlab(:);% ta bort det här
plot(filtered_y(1,:).' + 3*sqrt([fitt_matlab(:)]), '--b')
plot(filtered_y(1,:).' - 3*sqrt([fitt_matlab(:)]), '--b')
legend('pos', 'measured pos', 'filtered measure pos', '3 \sigma', '-3 \sigma')
title('b')


%plot vel
figure
plot(x(2,:))
hold on
plot(filtered_y(2,:))

fitt_matlab = cov(2,2,:);
fitt_matlab(:);% ta bort det här
plot(filtered_y(2,:).' + 3*sqrt([fitt_matlab(:)]), '--b')
plot(filtered_y(2,:).' - 3*sqrt([fitt_matlab(:)]), '--b')
legend('vel', 'filtered  vel', '3 \sigma', '-3 \sigma')
title('b')

%%
close all
% c
Q = @(t) [0 0; 0 t];
[filtered_y_1, cov1, Vk] = kalmanFilter(y, x0, P0, A, Q(0.1), H, R);
[filtered_y_10, cov10, Vk] = kalmanFilter(y, x0, P0, A, Q(1), H, R);
[filtered_y_100, cov100, Vk] = kalmanFilter(y, x0, P0, A, Q(10), H, R);
[filtered_y_15, cov15, Vk] = kalmanFilter(y, x0, P0, A, Q(1.5), H, R);

figure
plot(x(1,:).')
hold on
plot(filtered_y_1(1,:).')
plot(filtered_y_10(1,:).')
plot(filtered_y_100(1,:).')
plot(filtered_y_15(1,:).')
legend('x', 'i', 'ii', 'iii', 'iv')
title('vel')

figure
plot(x(2,:).')
hold on
plot(filtered_y_1(2,:).')
plot(filtered_y_10(2,:).')
plot(filtered_y_100(2,:).')
plot(filtered_y_15(2,:).')
legend('x', 'i', 'ii', 'iii', 'iv')
title('pos')

%%
plot_skalman(x,y,filtered_y_1, cov1, 'i')
plot_skalman(x,y,filtered_y_10, cov10, 'ii')
plot_skalman(x,y,filtered_y_100, cov100, 'iii')
plot_skalman(x,y,filtered_y_15, cov15, 'iv')


function plot_skalman(x, y, y_filtered ,cov, plot_title)
figure(1)
%plot(x(1,:))
hold on
%plot(y)
plot(y_filtered(1,:))

fitt_matlab = cov(1,1,:);
fitt_matlab(:);% ta bort det här
%plot([x(1,:).'] + 3*sqrt([fitt_matlab(:)]), '--b')
%plot([x(1,:).'] - 3*sqrt([fitt_matlab(:)]), '--b')
%legend('Predicted pos', 'filtered measure pos', '3 \sigma', '-3 \sigma')
title(plot_title)

%plot vel
figure(2)
plot(x(2,:))
hold on
plot(y_filtered(2,:))

fitt_matlab = cov(2,2,:);
fitt_matlab(:);% ta bort det här
%plot([x(2,:).'] + 3*sqrt([fitt_matlab(:)]), '--b')
%plot([x(2,:).'] - 3*sqrt([fitt_matlab(:)]), '--b')
%legend('Predicted vel', 'filtered predicted vel', '3 \sigma', '-3 \sigma')
title(plot_title)

end


function [X, P, Vks] = kalmanFilter(Y, x_0, P_0, A, Q, H, R)
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
Vks = [];


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
    [x, p, Vk] = linearUpdate(x, p, Y(:, i), H, R);           
    X = [X x];
    Vks = [Vks Vk];
    P(:,:,i) = p;
end
%%%%%%%%%%%%%%%%%
   
end

function [x, P, Vk] = linearUpdate(x, P, y, H, R)
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