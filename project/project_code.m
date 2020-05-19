clc; close all; clear;
 
startup()
%%
[xhat, meas] = filterTemplate();
%%
clc; clear all; close all;
[xhat, meas] = noa_filter();
%%
clc; close all; clear;

%load("meas_side.mat")
%load("meas_flat.mat")
%load("meas_turning.mat")

[xhat, meas] = noa_filter_no_bull_shit(meas);
euler_ang = quat2eul(xhat.x.', "XYZ");
plot(euler_ang)
legend('x', 'y', 'z')
 
%%
clc; close all; clear;
 
load("xhat_test")
load("meas_flat")
 
acc_mean = mean(meas.acc(:, ~any(isnan(meas.acc), 1)), 2);
gyr_mean = mean(meas.gyr(:, ~any(isnan(meas.gyr), 1)), 2);
mag_mean = mean(meas.mag(:, ~any(isnan(meas.mag), 1)), 2);
 
acc_cov = cov(meas.acc(:, ~any(isnan(meas.acc), 1)).');
gyr_cov = cov(meas.gyr(:, ~any(isnan(meas.gyr), 1)).');
mag_cov = cov(meas.mag(:, ~any(isnan(meas.mag), 1)).');
 
% Plot acc data
title('Acc readings')
acc = meas.acc(:, ~any(isnan(meas.acc), 1));
subplot(2,2,1)
histogram(acc(1,:))
title('acc(1)')
 
subplot(2,2,2)
histogram(acc(2,:))
title('acc(2)')
 
subplot(2,2,3)
histogram(acc(3,:))
title('acc(3)')
 
% Plot gyro data
figure
gyr = meas.gyr(:, ~any(isnan(meas.gyr), 1));
subplot(2,2,1)
histogram(gyr(1,:))
title('gyro(1)')
 
subplot(2,2,2)
histogram(gyr(2,:))
title('gyro(2)')
 
subplot(2,2,3)
histogram(gyr(3,:))
title('gyro(3)')
 
% Plot gyro data
figure
mag = meas.mag(:, ~any(isnan(meas.mag), 1));
subplot(2,2,1)
histogram(mag(1,:))
title('Mag(1)')
 
subplot(2,2,2)
histogram(mag(2,:))
title('Mag(2)')
 
subplot(2,2,3)
histogram(mag(3,:))
title('Mag(3)')
 
figure
plot(mag.')
hold on
plot([zeros(3,1), size(mag,2)*ones(3,1)].', [mag_mean mag_mean].', '-black')
legend('mag(1)', 'mag(2)', 'mag(3)', 'mag_{\mu}(1)', 'mag_{\mu}(2)', 'mag_{\mu}(3)')
 
figure
plot(gyr.')
hold on
plot([zeros(3,1), size(gyr,2)*ones(3,1)].', [gyr_mean gyr_mean].', '-black')
legend('gyr(1)', 'gyr(2)', 'gyr(3)', 'gyr_{\mu}(1)', 'gyr_{\mu}(2)', 'gyr_{\mu}(3)')
 
figure
plot(acc.')
hold on
plot([zeros(3,1), size(acc,2)*ones(3,1)].', [acc_mean acc_mean].')
legend('acc(1)', 'acc(2)', 'acc(3)', 'acc_{\mu}(1)', 'acc_{\mu}(2)', 'acc_{\mu}(3)')



%%
















