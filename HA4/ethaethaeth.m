clc; clear all; close all;

N = 5;
T = 0.1;

% Define prior
x_0     = [0]';
n       = length(x_0);
P_0     = diag(0);

% Covariance 
Q = 0;

% Motion model
motionModel = @cpIroven;


% Measurement noise covariance
R = 0;

% Measurement model
measModel = @fitt_matlab;

% function handle for generating sigma points
genSigmaPoints = @sigmaPoints;


X = 0:2:2*N;

% generate measurements
Y = 0:2:2*N;

% Kalman filter
S = zeros(2,6);
[xs, Ps, xf, Pf, xp, Pp] = nonLinRTSsmoother(Y, x_0, P_0, motionModel, T, Q, S, measModel, R, genSigmaPoints, 'EKF');

X = X(:, 2:end);

fprintf("xs(i)-x(i) = %d\n", xs-X);

%%
function [fitt, matlab] = fitt_matlab(x)
    fitt = x;
    matlab = 1;
end

function [kuken, horan] = cpIroven(x, t)
    kuken = 2*x;
    horan = 2;
end

function [SP,W] = sigmaPoints(x, P, type)
% SIGMAPOINTS computes sigma points, either using unscented transform or
% using cubature.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%
%Output:
%   SP          [n x 2n+1] matrix with sigma points
%   W           [1 x 2n+1] vector with sigma point weights 
%

    switch type        
        case 'UKF'

            % Dimension of state
            n = length(x);

            % Allocate memory
            SP = zeros(n,2*n+1);

            % Weights
            W = [1-n/3 repmat(1/6,[1 2*n])];

            % Matrix square root
            sqrtP = sqrtm(P);

            % Compute sigma points
            SP(:,1) = x;
            for i = 1:n
                SP(:,i+1) = x + sqrt(1/2/W(i+1))*sqrtP(:,i);
                SP(:,i+1+n) = x - sqrt(1/2/W(i+1+n))*sqrtP(:,i);
            end

        case 'CKF'

            % Dimension of state
            n = length(x);

            % Allocate memory
            SP = zeros(n,2*n);

            % Weights
            W = repmat(1/2/n,[1 2*n]);

            % Matrix square root
            sqrtP = sqrtm(P);

            % Compute sigma points
            for i = 1:n
                SP(:,i) = x + sqrt(n)*sqrtP(:,i);
                SP(:,i+n) = x - sqrt(n)*sqrtP(:,i);
            end

        otherwise
            error('Incorrect type of sigma point')
    end
end

function [h, H] = rangeBearingMeasurements(x, s)
%RANGEBEARINGMEASUREMENTS calculates the range and the bearing to the
%position given by the state vector x, from a sensor locateed in s
%
%Input:
%   x           [n x 1] State vector
%   s           [2 x 1] Sensor position
%
%Output:
%   h           [2 x 1] measurement vector
%   H           [2 x n] measurement model Jacobian
%
% NOTE: the measurement model assumes that in the state vector x, the first
% two states are X-position and Y-position.

    % Range
    rng = norm(x(1:2)-s);
    % Bearing
    ber = atan2(x(2)-s(2),x(1)-s(1));
    % Measurement vector
    h = [rng;ber];

    % Measurement model Jacobian
    H = [
        (x(1)-s(1))/rng      (x(2)-s(2))/rng     0 0 0;
        -(x(2)-s(2))/(rng^2) (x(1)-s(1))/(rng^2) 0 0 0
        ];

end

function [f, F] = coordinatedTurnMotion(x, T)
%COORDINATEDTURNMOTION calculates the predicted state using a coordinated
%turn motion model, and also calculated the motion model Jacobian
%
%Input:
%   x           [5 x 1] state vector
%   T           [1 x 1] Sampling time
%
%Output:
%   f           [5 x 1] predicted state
%   F           [5 x 5] motion model Jacobian
%
% NOTE: the motion model assumes that the state vector x consist of the
% following states:
%   px          X-position
%   py          Y-position
%   v           velocity
%   phi         heading
%   omega       turn-rate

    % Velocity
    v = x(3);
    % Heading
    phi = x(4);
    % Turn-rate
    omega = x(5);

    % Predicted state
    f = x + [
        T*v*cos(phi);
        T*v*sin(phi);
        0;
        T*omega;
        0];

    % Motion model Jacobian
    F = [
        1 0 T*cos(phi) -T*v*sin(phi) 0;
        0 1 T*sin(phi) T*v*cos(phi)  0;
        0 0 1          0             0;
        0 0 0          1             T;
        0 0 0          0             1
        ];
end

function X = genNonLinearStateSequence(x_0, P_0, f, T, Q, N)
%GENLINEARSTATESEQUENCE generates an N-long sequence of states using a 
%    Gaussian prior and a linear Gaussian process model
%
%Input:
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   f           Motion model function handle
%   T           Sampling time
%   Q           [n x n] Process noise covariance
%   N           [1 x 1] Number of states to generate
%
%Output:
%   X           [n x N] State vector sequence
%

    % Dimension of state vector
    n = length(x_0);

    % allocate memory
    X = zeros(n, N);

    % Generete start state
    X(:,1) = mvnrnd(x_0', P_0)';

    % Generate sequence
    for k = 2:N+1

        % generate noise vector
        q = mvnrnd(zeros(1,n), Q)';

        % Propagate through process model
        [fX, ~] = f(X(:,k-1),T);
        X(:,k) = fX + q;

    end

end

function Y = genNonLinearMeasurementSequence(X, S, h, R)
%GENNONLINEARMEASUREMENTSEQUENCE generates ovservations of the states 
% sequence X using a non-linear measurement model.
%
%Input:
%   X           [n x N+1] State vector sequence
%   S           [n x N] Sensor position vector sequence
%   h           Measurement model function handle
%   R           [m x m] Measurement noise covariance
%
%Output:
%   Y           [m x N] Measurement sequence
%

    % Parameters
    N = size(X,2);
    m = size(R,1);

    % Allocate memory
    Y = zeros(m,N-1);

    for k = 1:N-1
        % Measurement
        [hX,~] = h(X(:,k+1),S(:,k));
        % Add noise
        Y(:,k) = hX + mvnrnd(zeros(1,m), R)';

    end

end

%%
function [xs, Ps, xf, Pf, xp, Pp] = nonLinRTSsmoother(Y, x_0, P_0, f, T, Q, S, h, R, sigmaPoints, type)
%NONLINRTSSMOOTHER Filters measurement sequence Y using a 
% non-linear Kalman filter. 
%
%Input:
%   Y           [m x N] Measurement sequence for times 1,...,N
%   x_0         [n x 1] Prior mean for time 0
%   P_0         [n x n] Prior covariance
%   f                   Motion model function handle
%   T                   Sampling time
%   Q           [n x n] Process noise covariance
%   S           [n x N] Sensor position vector sequence
%   h                   Measurement model function handle
%   R           [n x n] Measurement noise covariance
%   sigmaPoints Handle to function that generates sigma points.
%   type        String that specifies type of non-linear filter/smoother
%
%Output:
%   xf          [n x N]     Filtered estimates for times 1,...,N
%   Pf          [n x n x N] Filter error convariance
%   xp          [n x N]     Predicted estimates for times 1,...,N
%   Pp          [n x n x N] Filter error convariance
%   xs          [n x N]     Smoothed estimates for times 1,...,N
%   Ps          [n x n x N] Smoothing error convariance

% your code here!
% We have offered you functions that do the non-linear Kalman prediction and update steps.
% Call the functions using
% [xPred, PPred] = nonLinKFprediction(x_0, P_0, f, T, Q, sigmaPoints, type);
% [xf, Pf] = nonLinKFupdate(xPred, PPred, Y, S, h, R, sigmaPoints, type);



N = size(Y,2);

n = length(x_0);
m = size(Y,1);

% Data allocation
Pp = zeros(n,n,N);
Pf = zeros(n,n,N);
Ps = zeros(n,n,N);

% Start with going forward
% Predict one step ahead in time.
[x, p] = nonLinKFprediction(x_0, P_0, f, T, Q, sigmaPoints, type);
xp = x;
Pp(:,:,1) = p;


% Update the estimated position with the measurement
[x, p] = nonLinKFupdate(x, p, Y(:, 1), S(:,1), h, R, sigmaPoints, type);
xf = x;
Pf(:,:,1) = p;
    
for i = 2:size(Y,2)
    % Predict one step ahead in time.
    [x, p] = nonLinKFprediction(xf(:, end), Pf(:,:, i-1), f, T, Q, sigmaPoints, type);
    xp = [xp x];
    Pp(:,:,i) = p;
    
    % Update the estimated position with the measurement
    [x, p] = nonLinKFupdate(x, p, Y(:, i), S(:,i), h, R, sigmaPoints, type);
    xf = [xf x];
    Pf(:,:,i) = p;
end
% Done with forward

% Start going backwards
xs = xf(:, end);
Ps(:,:,end) = Pf(:,:,end);
for i = N-1:-1:1
    [x_smoothed, P_smoothed] = nonLinRTSSupdate(xs(:,1), Ps(:,:,i+1), xf(:,i), Pf(:,:,i), xp(:,i+1), Pp(:,:,i+1), f, T, sigmaPoints, type);
    xs = [x_smoothed xs];
    Ps(:,:,i) = P_smoothed;
end

if type == 'EKF'
    % Pf(:,:,end-1)
    % Ps(:,:,end-1)
    xs(:, end-1);
    xf(:, end-1);
    xs
end
%xs = [flip(xs(1,:)); flip(xs(2,:)); flip(xs(3,:)); flip(xs(4,:)); flip(xs(5,:)) ];

end

function [xs, Ps] = nonLinRTSSupdate(xs_kplus1, Ps_kplus1,  xf_k,  Pf_k, xp_kplus1, Pp_kplus1, f, T, sigmaPoints, type)
%NONLINRTSSUPDATE Calculates mean and covariance of smoothed state
% density, using a non-linear Gaussian model.
%
%Input:
%   xs_kplus1   Smooting estimate for state at time k+1
%   Ps_kplus1   Smoothing error covariance for state at time k+1
%   xf_k        Filter estimate for state at time k
%   Pf_k        Filter error covariance for state at time k
%   xp_kplus1   Prediction estimate for state at time k+1
%   Pp_kplus1   Prediction error covariance for state at time k+1
%   f           Motion model function handle
%   T           Sampling time
%   sigmaPoints Handle to function that generates sigma points.
%   type        String that specifies type of non-linear filter/smoother
%
%Output:
%   xs          Smoothed estimate of state at time k
%   Ps          Smoothed error convariance for state at time k

% Your code here.
if type == "EKF"
    % Calculate the differentiation of the state in xf_k
    [x_kp1 dx] = f(xf_k,T);
    
    % Calculate the smoothed state and smoothed covariance
    G = Pf_k * dx * inv(Pp_kplus1);
    xs = xf_k + G*(xs_kplus1 - x_kp1);
    Ps = Pf_k - G*(Pp_kplus1 - Ps_kplus1)*G.';

elseif type == "UKF"  | type == "CKF"
    % Calculate the differentiation of the state in xf_k
    [sp, W] = sigmaPoints(xf_k, Pf_k, type);
    x_kp1 = f(sp, T)*W.';
    
    P_kkp1 = zeros(size(Pp_kplus1));
    for i = 1 : size(W,2)
        P_kkp1 = P_kkp1 + (sp(:,i)-xf_k)*(f(sp(:,i), T)-xp_kplus1).'*W(i);
    end
    
    % Calculate the smoothed state and smoothed covariance
    G = P_kkp1*inv(Pp_kplus1);
    xs = xf_k + G*(xs_kplus1 - x_kp1);
    Ps = Pf_k - G*(Pp_kplus1 - Ps_kplus1)*G.';
    
    
end


end

function [x, P, S, y_pred] = nonLinKFupdate(x, P, y, sens, h, R, sigmaPoints, type)
%NONLINKFUPDATE calculates mean and covariance of predicted state
%   density using a non-linear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   y           [m x 1] measurement vector
%   h           Measurement model function handle
%               [hx,Hx]=h(x) 
%               Takes as input x (state), 
%               Returns hx and Hx, measurement model and Jacobian evaluated at x
%               Function must include all model parameters for the particular model, 
%               such as sensor position for some models.
%   R           [m x m] Measurement noise covariance
%   type        String that specifies the type of non-linear filter
%
%Output:
%   x           [n x 1] updated state mean
%   P           [n x n] updated state covariance
%

    switch type
        case 'EKF'
            
            % Your EKF update here
            %%%%%%%%%%%%%%%%%%%%%%
            [hx,Hx]=h(x);
            y_pred = hx;
            S = Hx*P*Hx.' + R; % Predict the covariance in yk
            K = P*Hx.'*S^-1;  % Calculate the Kalman gain, how much we trust the new measurement
            P = P - K*S*K.'; %Estimate the error covariance
            x = x + K*(y-hx); % Estimate the new state
            %%%%%%%%%%%%%%%%%%%%%%
            
        case 'UKF'
    
            % Your UKF update here
            %%%%%%%%%%%%%%%%%%%%%
            [SP,W] = sigmaPoints(x, P, type);
            % Predict y
            y_pred = 0;
            for i = 1 : size(W,2)
                [hx,Hx] = h(SP(:, i));
                y_pred = y_pred + hx*W(i);
            end
            % Estimate x covariance
            Pxy = 0;
            S = 0;
            for i = 1 : size(W,2)
                [hx, Hx] = h(SP(:, i));
                Pxy = Pxy + ((SP(:,i) - x)*(hx - y_pred).')*W(i);
                S = S + (hx - y_pred)*(hx - y_pred).'*W(:,i);
            end
            S = S + R;
            P = P - Pxy*S^-1*Pxy.';
            
            % Estimate x
            x = x + Pxy*S^-1*(y-y_pred);
            %%%%%%%%%%%%%%%%%%%%%
            
            % Make sure the covariance matrix is semi-definite
            if min(eig(P))<=0
                [v,e] = eig(P, 'vector');
                e(e<0) = 1e-4;
                P = v*diag(e)/v;
            end
            
        case 'CKF'
    
            % Your CKF update here
            %%%%%%%%%%%%%%%%%%%%%
            [SP,W] = sigmaPoints(x, P, type);
            % Predict y
            y_pred = 0;
            for i = 1 : size(W,2)
                [hx,Hx] = h(SP(:, i));
                y_pred = y_pred + hx*W(i);
            end
            
            % Estimate x covariance
            Pxy = 0;
            S = 0;
            for i = 1 : size(W,2)
                [hx, Hx] = h(SP(:, i));
                Pxy = Pxy + ((SP(:,i) - x)*(hx - y_pred).')*W(i);
                S = S + (hx - y_pred)*(hx - y_pred).'*W(:,i);
            end
            S = S + R;
            P = P - Pxy*S^-1*Pxy.';
            
            % Estimate x
            x = x + Pxy*S^-1*(y-y_pred);
            %%%%%%%%%%%%%%%%%%%%%
        otherwise
            error('Incorrect type of non-linear Kalman filter')
    end

end

function [x, P] = nonLinKFprediction(x, P, f, T, Q, sigmaPoints, type)
%NONLINKFPREDICTION calculates mean and covariance of predicted state
%   density using a non-linear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   f           Motion model function handle
%               [fx,Fx]=f(x) 
%               Takes as input x (state), 
%               Returns fx and Fx, motion model and Jacobian evaluated at x
%               All other model parameters, such as sample time T,
%               must be included in the function
%   Q           [n x n] Process noise covariance
%   type        String that specifies the type of non-linear filter
%
%Output:
%   x           [n x 1] predicted state mean
%   P           [n x n] predicted state covariance
%

f = @(x)f(x,T);

    switch type
        case 'EKF'
            
            % Your EKF code here
            %%%%%%%%%%%%%%%%%%%%
            % Prediction step
            [fx,Fx]=f(x);
            % x mean
            x = fx;
            % x covariance
            P = Fx*P*Fx.' + Q;
            %%%%%%%%%%%%%%%%%%%%
        case 'UKF'
    
            % Your UKF code here
            %%%%%%%%%%%%%%%%%%%%
            [SP,W] = sigmaPoints(x, P, type);
            
            % Predict x mean
            x = 0;
            P = 0;
            for i = 1 : size(W,2)
                [fx,Fx]=f(SP(:, i));
                x = x + fx*W(i);
            end
            % Predict x covariance
            for i = 1 : size(W,2)
                [fx,Fx]=f(SP(:, i));
                P = P + ((fx-x)*(fx-x).')*W(i);
            end
            P = P + Q;
            %%%%%%%%%%%%%%%%%%%%
        
            % Make sure the covariance matrix is semi-definite
            if min(eig(P))<=0
                [v,e] = eig(P, 'vector');
                e(e<0) = 1e-4;
                P = v*diag(e)/v;
            end
                
        case 'CKF'
            
            % Your CKF code here
            %%%%%%%%%%%%%%%%%%%%
            [SP,W] = sigmaPoints(x, P, type);
            
            % Predict x mean
            x = 0;
            P = 0;
            for i = 1 : size(W,2)
                [fx,Fx]=f(SP(:, i));
                x = x + fx*W(i);
            end
            % Predict x covariance
            for i = 1 : size(W,2)
                [fx,Fx]=f(SP(:, i));
                P = P + ((fx-x)*(fx-x).')*W(i);
            end
            P = P + Q;
            %%%%%%%%%%%%%%%%%%%%
        otherwise
            error('Incorrect type of non-linear Kalman filter')
    end

end

