function [xhat, meas] = noa_filter_no_bull_shit(meas)
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

    % Used for visualization.
    figure(1);
    subplot(1, 2, 1);
    ownView = OrientationView('Own filter', gca);  % Used for visualization.
    googleView = [];
    counter = 0;  % Used to throttle the displayed frame rate.

    % Filter settings
    t0 = [];  % Initial time (initialize on first data received)
    nx = 4;   % Assuming that you use q as state variable.
    
    % Add your filter settings here.

    Rw = 1e-5 * [0.081733901510542   0.001723173871440  -0.002582723004641
                 0.001723173871440   0.105466716020954   0.000782168729699
                 -0.002582723004641   0.000782168729699   0.061062268681170];
    
    g0 = -[0.0171 -0.0403 -9.9719].';
    Ra =    1.0e-03 *[0.1356   -0.0005    0.0001
                      -0.0005    0.1441   -0.0066
                      0.0001   -0.0066    0.2921];
             
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
        [x, P] = mu_g(x, P, acc, Ra, g0);
    end
    
    gyr = meas.gyr(:,i);
    if ~any(isnan(gyr))  % Gyro measurements are available.
        [x,P] = tu_qw(x, P, gyr, t-meas.t(i-1), Rw); 
    elseif ~any(isnan(meas.gyr(:, i)))
          [x,P] = tu_qw(x, P, gyr, t-t0-meas.t(end), Rw);
    end

    mag = meas.mag(:,i);
    if ~any(isnan(mag))  % Mag measurements are available.
        % Do something
    end
    
    orientation = meas.orient(:,i)';  % Google's orientation estimate.

      % Visualize result
      if rem(counter, 10) == 0
        setOrientation(ownView, x(1:4));
        title(ownView, 'OWN', 'FontSize', 16);
        if ~any(isnan(orientation))
          if isempty(googleView)
            subplot(1, 2, 2);
            % Used for visualization.
            googleView = OrientationView('Google filter', gca);
          end
          setOrientation(googleView, orientation);
          title(googleView, 'GOOGLE', 'FontSize', 16);
        end
      end
      counter = counter + 1;
    
    % Save estimates
    xhat.x(:, end+1) = x;
    xhat.P(:, :, end+1) = P;
    xhat.t(end+1) = t - t0;
    old_t = t;
    
    end
end







