function [xhat, meas] = filterTemplate(meas)
% FILTERTEMPLATE  Filter template
%
% This is a template function for how to collect and filter data
% sent from a smartphone live.  Calibration data for the
% accelerometer, gyroscope and magnetometer assumed available as
% structs with fields m (mean) and R (variance).
%
% The function returns xhat as an array of structs comprising t
% (timestamp), x (state), and P (state covariance) for each
% timestamp, and meas an array of structs comprising t (timestamp),
% acc (accelerometer measurements), gyr (gyroscope measurements),
% mag (magnetometer measurements), and orint (orientation quaternions
% from the phone).  Measurements not availabe are marked with NaNs.
%
% As you implement your own orientation estimate, it will be
% visualized in a simple illustration.  If the orientation estimate
% is checked in the Sensor Fusion app, it will be displayed in a
% separate view.
%
% Note that it is not necessary to provide inputs (calAcc, calGyr, calMag).

    % Filter settings
    t0 = [];  % Initial time (initialize on first data received)
    nx = 4;   % Assuming that you use q as state variable.
    
    % Add your filter settings here.

    Rw = 1e-5 * [0.081733901510542   0.001723173871440  -0.002582723004641
                 0.001723173871440   0.105466716020954   0.000782168729699
                 -0.002582723004641   0.000782168729699   0.061062268681170];
             
    t0 = meas.t(1);
    old_t = t0;
    
    % Current filter state.
    x = [1; 0; 0 ;0];
    P = eye(nx, nx);

    % Saved filter states.
    xhat = struct('t', zeros(1, 0),...
                  'x', zeros(nx, 0),...
                  'P', zeros(nx, nx, 0));

    
    i = 1;
    % Filter loop
    while i < size(meas.t, 2)  % Repeat while data is available
    i = i + 1;
    t = meas.t(i);
      
    acc = meas.acc(:,i);
    if ~any(isnan(acc))  % Acc measurements are available.
        % Do something
    end
    
    gyr = meas.gyr(:,i);
    if ~any(isnan(gyr))  % Gyro measurements are available.
        %[x,P] = tu_qw(x, P, gyr, t-meas.t(end), Rw); %WTF is P? I never
        %do anything with P OMG WTF IS THIS
        [x,P] = tu_qw(x, 0, gyr, t-meas.t(i-1), Rw);
    end

    mag = meas.mag(:,i);
    if ~any(isnan(mag))  % Mag measurements are available.
        % Do something
    end
    
    % Save estimates
    xhat.x(:, end+1) = x;
    xhat.P(:, :, end+1) = P;
    xhat.t(end+1) = t - t0;
    old_t = t;
    
    end
end

function [x,P] = tu_qw(x, P, omega, T, Rw)
    % Not sure what to use P for
    % Why should I handle the case if there is no omega? Nothing happens if
    % there is no omega anyway
    % WTF is x? Is that q? If not how the fuck do I get q????
    gyr_noise_mean = 1e-3 * [-0.1026; 0.2574; 0.0012];
    x = 1/2 * ((eye(size(x)) + Somega(omega)*T)*x + (eye(size(x)) + Sq(x)*T)*mvnrnd(gyr_noise_mean, Rw).');
    [x, P] = mu_normalizeQ(x, P);
end
