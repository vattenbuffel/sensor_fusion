clc; clear all; close all;


% ToDo? Check the dimensions of the outputs

genSigmaPoints = @sigmaPoints;
tol = 1e-4;
for k = 1:100
    % Process model
    f = @linProcModel;
    T = 1;

    % filter state k|k
    xf_k = 20*rand(2,1);
    % filter Covariance k|k
    Pf_k = sqrt(10)*rand(2,2);
    Pf_k = Pf_k*Pf_k';

    % predicted state k+1|k
        xp_kplus1 = f(xf_k);

    % prediction Covariance k+1|k
        % Increase Pp_k by some factor
        Pp_kplus1 = Pf_k*10;

    % smoothed state k+1|K
        % move in random direction, a fraction of the distance between
        % x_k+1|k and x_k|k
        dir = rand(2,1);
        dir = dir/norm(dir);

        dist = norm(xp_kplus1-xf_k); % distance between filter estimate and prediction 

        xs_kplus1 = xp_kplus1 + dir*dist*0.3;

    % smoothing covariance k+1|K
        % Decrease Pp_k by some factor
        Ps_kplus1 = Pf_k*0.5;

    % EKF
    [xs, Ps] =         nonLinRTSSupdate(xs_kplus1, Ps_kplus1, xf_k, Pf_k, ...
                                        xp_kplus1, Pp_kplus1, f, T, genSigmaPoints, 'EKF');
    
    % Test results
    
    % UKF
    [xsu, Ps] =         nonLinRTSSupdate(xs_kplus1, Ps_kplus1, xf_k, Pf_k, ...
                                        xp_kplus1, Pp_kplus1, f, T, genSigmaPoints, 'UKF');
    
    % CKF
    [xsc, Ps] =         nonLinRTSSupdate(xs_kplus1, Ps_kplus1, xf_k, Pf_k, ...
                                        xp_kplus1, Pp_kplus1, f, T, genSigmaPoints, 'CKF');
    break

end

function [xk, J] = linProcModel(xkmin1, T)

    A = [1 0; 0 1];
    b = [1 1]';
    xk = A*xkmin1 + b;
    
    J = A;
end


%% new functions

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
X_k = proc_f(X_kmin1) + mvnrnd(zeros(size(X_kmin1, 1), 1) , (proc_Q), size(X_kmin1, 2)).';

W_k = zeros(size(W_kmin1));
for i = 1:size(X_kmin1, 2)
   W_k(i) = W_kmin1(i) * normpdf(yk, meas_h(X_k(:, i)), sqrtm(meas_R));
end

W_k = W_k / sum(W_k);
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
xp = [];
xf = [];
x = x_0;
p = P_0;
 for i = 1:size(Y,2)
    % Predict one step ahead in time.
    [x, p] = nonLinKFprediction(x, p, f, T, Q, sigmaPoints, type);
    xp = [xp x];
    Pp(:,:,i) = p;
    
    % Update the estimated position with the measurement
    [x, p] = nonLinKFupdate(x, p, Y(:, i), S(:,i), h, R, sigmaPoints, type);
    xf = [xf x];
    Pf(:,:,i) = p;
end
% Done with forward


% Start going backwards
% The last smoothed state is the same as the last filtered state
xs = xf(:,end);
Ps(:,:,end) = Pf(:,:,end);
for i = N-1:-1:1 % Smooth backwards one state at a time
    [x_smoothed, P_smoothed] = nonLinRTSSupdate(xs(:,1), Ps(:,:,i+1), xf(:,i), Pf(:,:,i), xp(:,i+1), Pp(:,:,i+1), f, T, sigmaPoints, type);

    xs = [x_smoothed xs];
    Ps(:,:,i) = P_smoothed;
end

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
    [x_kp1, dx] = f(xf_k,T);
    
    % Calculate the smoothed state and smoothed covariance
    G = Pf_k * dx.' * inv(Pp_kplus1);
    xs = xf_k + G*(xs_kplus1 - xp_kplus1);
    Ps = Pf_k - G*(Pp_kplus1 - Ps_kplus1)*G.';

elseif type == "UKF"  | type == "CKF"
    % Calculate the differentiation of the state in xf_k
    [sp, W] = sigmaPoints(xf_k, Pf_k, type);
    %x_kp1 = f(sp, T)*W.';
    
    P_kkp1 = zeros(size(Pp_kplus1));
    for i = 1 : size(W,2)
        P_kkp1 = P_kkp1 + (sp(:,i)-xf_k)*(f(sp(:,i), T)-xp_kplus1).'*W(i);
    end
    
    % Calculate the smoothed state and smoothed covariance
    G = P_kkp1*inv(Pp_kplus1);
    xs = xf_k + G*(xs_kplus1 - xp_kplus1);
    Ps = Pf_k - G*(Pp_kplus1 - Ps_kplus1)*G.';
    
    
end


end







%% old functions
function [xf, Pf, xp, Pp] = nonLinearKalmanFilter(Y, x_0, P_0, f, Q, h, R, type)
%NONLINEARKALMANFILTER Filters measurement sequence Y using a 
% non-linear Kalman filter. 
%
%Input:
%   Y           [m x N] Measurement sequence for times 1,...,N
%   x_0         [n x 1] Prior mean for time 0
%   P_0         [n x n] Prior covariance
%   f                   Motion model function handle
%                       [fx,Fx]=f(x) 
%                       Takes as input x (state) 
%                       Returns fx and Fx, motion model and Jacobian evaluated at x
%   Q           [n x n] Process noise covariance
%   h                   Measurement model function handle
%                       [hx,Hx]=h(x,T) 
%                       Takes as input x (state), 
%                       Returns hx and Hx, measurement model and Jacobian evaluated at x
%   R           [m x m] Measurement noise covariance
%
%Output:
%   xf          [n x N]     Filtered estimates for times 1,...,N
%   Pf          [n x n x N] Filter error convariance
%   xp          [n x N]     Predicted estimates for times 1,...,N
%   Pp          [n x n x N] Filter error convariance
%

% Your code here. If you have good code for the Kalman filter, you should re-use it here as
% much as possible.

% My code
%%%%%%%%%%%%%%%%%
% Parameters
N = size(Y,2);

n = length(x_0);
m = size(Y,1);

% Data allocation
Pp = zeros(n,n,N);
Pf = zeros(n,n,N);

% Predict one step ahead in time.
[x, p] = nonLinKFprediction(x_0, P_0, f, Q, type);
xp = x;
Pp(:,:,1) = p;

% Update the estimated position with the measurement
[x, p] = nonLinKFupdate(x, p, Y(:, 1), h, R, type);
xf = x;
Pf(:,:,1) = p;
    
for i = 2:size(Y,2)
    % Predict one step ahead in time.
    [x, p] = nonLinKFprediction(xf(:, end), Pf(:,:, i-1), f, Q, type);
    xp = [xp x];
    Pp(:,:,i) = p;
    
    % Update the estimated position with the measurement
    [x, p] = nonLinKFupdate(x, p, Y(:, i), h, R, type);
    xf = [xf x];
    Pf(:,:,i) = p;
end
%%%%%%%%%%%%%%%%%
   

end

function [x, P, S, y_pred] = nonLinKFupdate(x, P, y, h, R, type)
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

function [x, P] = nonLinKFprediction(x, P, f, Q, type)
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

function [SP,W] = sigmaPoints(x, P, type)
% SIGMAPOINTS computes sigma points, either using unscented transform or
% using cubature.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%
%Output:
%   SP          [n x 2n+1] UKF, [n x 2n] CKF. Matrix with sigma points
%   W           [1 x 2n+1] UKF, [1 x 2n] UKF. Vector with sigma point weights 
%
    switch type        
        case 'UKF'
    
            % your code
            %%%%%%%%%%%%%%%
            n = size(x, 1);
            
            % Calculate the SP weights
            w0 = 1-n/3;
            W = [w0 (1-w0)/(2*n)*ones(1, 2*n)];
            
            % Generate the SP locations/values
            P_sqrt = sqrtm(P);
            SP = [x...
                x+sqrt(n/(1-w0))*P_sqrt...
                x-sqrt(n/(1-w0))*P_sqrt];
            %%%%%%%%%%%%%%%
                
        case 'CKF'
            
            % your code
            %%%%%%%%%%%%%%%
            n = size(x, 1);
            
            % Generate the SP locations/values
            P_sqrt = sqrtm(P);
            SP = [x+sqrt(n)*P_sqrt, x-sqrt(n)*P_sqrt];
            
            % Calculate the SP weights
            W = 1/(2*n)*ones(1, 2*n);
            %%%%%%%%%%%%%%%
            
        otherwise
            error('Incorrect type of sigma point')
    end

end

function Y = genNonLinearMeasurementSequence(X, h, R)
%GENNONLINEARMEASUREMENTSEQUENCE generates ovservations of the states 
% sequence X using a non-linear measurement model.
%
%Input:
%   X           [n x N+1] State vector sequence
%   h           Measurement model function handle
%               [hx,Hx]=h(x) 
%               Takes as input x (state) 
%               Returns hx and Hx, measurement model and Jacobian evaluated at x
%   R           [m x m] Measurement noise covariance
%
%Output:
%   Y           [m x N] Measurement sequence
%

% Your code here
%%%%%%%%%%%%%%%%%%
% Remove the first state as we don't want to measure that
X = X(:, 2:end);
Y = [];
for i = 1:size(X,2)
   % Measure the next state
   y = h(X(:, i));
   % Add noise to the measurement
   y = y + mvnrnd(zeros(size(y, 1), 1), R).';
   % Save the measurement
   Y = [Y y]; 
end
%%%%%%%%%%%%%%%%%%

end

function X = genNonLinearStateSequence(x_0, P_0, f, Q, N)
%GENLINEARSTATESEQUENCE generates an N+1-long sequence of states using a 
%    Gaussian prior and a linear Gaussian process model
%
%Input:
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   f           Motion model function handle
%               [fx,Fx]=f(x) 
%               Takes as input x (state), 
%               Returns fx and Fx, motion model and Jacobian evaluated at x
%               All other model parameters, such as sample time T,
%               must be included in the function
%   Q           [n x n] Process noise covariance
%   N           [1 x 1] Number of states to generate
%
%Output:
%   X           [n x N+1] State vector sequence
%

% Your code here
% Generate first state
X = mvnrnd(x_0, P_0).';
for i=1:N
    % Generate next state
    [fx,Fx]=f(X(:, end));
    % Append the new state to the state vector and apply motion noise to it
    X = [X fx+mvnrnd(zeros(size(Q,2), 1), Q).'];
end

end

function [hx, Hx] = dualBearingMeasurement(x, s1, s2)
%DUOBEARINGMEASUREMENT calculates the bearings from two sensors, located in 
%s1 and s2, to the position given by the state vector x. Also returns the
%Jacobian of the model at x.
%
%Input:
%   x           [n x 1] State vector, the two first element are 2D position
%   s1          [2 x 1] Sensor position (2D) for sensor 1
%   s2          [2 x 1] Sensor position (2D) for sensor 2
%
%Output:
%   hx          [2 x 1] measurement vector
%   Hx          [2 x n] measurement model Jacobian
%
% NOTE: the measurement model assumes that in the state vector x, the first
% two states are X-position and Y-position.

% Your code here
% Calculate y+, without the noise
hx = [atan2(x(2,:)-s1(2), x(1,:)-s1(1))
      atan2(x(2,:)-s2(2), x(1,:)-s2(1))];
  
% Calculate deriv(y+) evaluated at x, without the noise
Hx = [-(x(2,:)-s1(2,:))./((x(2,:)-s1(2,:)).^2 + (x(1,:)-s1(1,:)).^2) (x(1,:)-s1(1,:))./((x(2,:)-s1(2,:)).^2 + (x(1,:)-s1(1,:)).^2)
      -(x(2,:)-s2(2,:))./((x(2,:)-s2(2,:)).^2 + (x(1,:)-s2(1,:)).^2) (x(1,:)-s2(1,:))./((x(2,:)-s2(2,:)).^2 + (x(1,:)-s2(1,:)).^2)];
% Append zeros to the end cuz the rest of the states don't depent on x(1) or x(2)
Hx = [Hx zeros(2, size(x,1)-size(Hx,1))];
end

function [fx, Fx] = coordinatedTurnMotion(x, T)
%COORDINATEDTURNMOTION calculates the predicted state using a coordinated
%turn motion model, and also calculated the motion model Jacobian
%
%Input:
%   x           [5 x 1] state vector
%   T           [1 x 1] Sampling time
%
%Output:
%   fx          [5 x 1] motion model evaluated at state x
%   Fx          [5 x 5] motion model Jacobian evaluated at state x
%
% NOTE: the motion model assumes that the state vector x consist of the
% following states:
%   px          X-position
%   py          Y-position
%   v           velocity
%   phi         heading
%   omega       turn-rate

% Your code for the motion model here
% Calculate the next state vector x, dissregarding the noise:
% x+ = [x+, y+, v+, theta+, omega+]
fx = [x(1) + T*x(3)*cos(x(4))
      x(2) + T*x(3)*sin(x(4))
      x(3)
      x(4) + T*x(5)
      x(5)];

%Check if the Jacobian is requested by the calling function
if nargout > 1
    % Your code for the motion model Jacobian here
    % F(x) is the derivation of x+ with respect to x evaluated at xk
    Fx = [1 0 T*cos(x(4)) -T*x(3)*sin(x(4)) 0
          0 1 T*sin(x(4)) T*x(3)*cos(x(4)) 0
          0 0 1 0 0
          0 0 0 1 T
          0 0 0 0 1];
end

end