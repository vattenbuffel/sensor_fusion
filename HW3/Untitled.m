clc; clear all; close all;

x = 5;
y = 5;

phi1 = atan2(y, x);
phi2 = atan2(y, -5);

t1 = tan(phi1);
t2 = tan(phi2);

s1y = 0;
s1x = 0;
s2y = 0;
s2x = 10;

(t2*(-s1y+s1x*t1-s2x*t1) + t1*s2y)/(t1-t2)