syms q1 q2 q3 q4
q = [q1;q2;q3;q4];

g0 = -[0.0171 -0.0403 -9.9719].';

h = Qq(q).'*g0;
h_der = jacobian(h, q)


%%
clc; clear all; close all
x = eye(4,1);
q2euler(x);
T = 1;
omega = [1;0;0];

F = (eye(size(x)) + T/2*Somega(omega));
x1 = F*x;

quat2eul(x1.', 'XYZ').'