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
    % Why should I handle the case if there is no omega? Nothing happens if
    % there is no omega anyway
    
    x = (eye(size(x)) + T*Somega(omega))/2*x;
    
    wx = omega(1);
    wy = omega(2);
    wz = omega(3);
    dfx = [1/2, 1/2 - (T*wx)/2, 1/2 - (T*wy)/2, 1/2 - (T*wz)/2
          (T*wx)/2,              0,       (T*wz)/2,      -(T*wy)/2
          (T*wy)/2,      -(T*wz)/2,              0,       (T*wx)/2
          (T*wz)/2,       (T*wy)/2,      -(T*wx)/2,              0];
    
    
%     %Used when calculating dfv
%     syms v1 v2 v3  T q1 q2 q3 q4
%     v = [v1;v2;v3];
%     q = [q1;q2;q3;q4];
%     dfv = T/2*Sq(q)*v;
%     dfv = jacobian(dfv, v);
    
    v1 = 0;
    v2 = 0;
    v3 = 0;
    q1 = x(1);
    q2 = x(2);
    q3 = x(3);
    q4 = x(4); 
    dfv =   [-(T*q2)/2, -(T*q3)/2, -(T*q4)/2
            (T*q1)/2, -(T*q4)/2,  (T*q3)/2
            (T*q4)/2,  (T*q1)/2, -(T*q2)/2
            -(T*q3)/2,  (T*q2)/2,  (T*q1)/2];
    
    P = dfx*P*dfx.' + dfv*Rw*dfv.'; 
    [x, P] = mu_normalizeQ(x, P);
end


function [x, P] = mu_g(x, P, yacc, Ra, g0)
    h = Qq(x).'*g0;

    % Used when calculating h_der
%     syms q1 q2 q3 q4
%     q = [q1;q2;q3;q4];
%     h_der = Qq(q).'*g0;
%     h_der = jacobian(h_der, q);
%     h_der = subs(h_der, q, x);
    %
    q1 = x(1);
    q2 = x(2);
    q3 = x(3);
    q4 = x(4);
    h_der =  [(403*q4)/5000 - (99719*q3)/5000 - (171*q1)/2500, (403*q3)/5000 - (171*q2)/2500 + (99719*q4)/5000,                 (403*q2)/5000 - (99719*q1)/5000,                 (403*q1)/5000 + (99719*q2)/5000
 (403*q1)/2500 + (99719*q2)/5000 + (171*q4)/5000,                 (99719*q1)/5000 - (171*q3)/5000, (403*q3)/2500 - (171*q2)/5000 + (99719*q4)/5000,                 (171*q1)/5000 + (99719*q3)/5000
 (99719*q1)/2500 - (403*q2)/5000 - (171*q3)/5000,                 - (403*q1)/5000 - (171*q4)/5000,                   (403*q4)/5000 - (171*q1)/5000, (403*q3)/5000 - (171*q2)/5000 + (99719*q4)/2500];
    
    S = h_der*P*h_der.' + Ra;
    K = P*h_der.'*S^-1;
    x = x + K*(yacc-h);
    P = P - K*S*K.';
end








