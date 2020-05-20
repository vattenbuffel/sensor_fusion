function [x,P] = tu_qw(x, P, omega, T, Rw)
    % Why should I handle the case if there is no omega? Nothing happens if
    % there is no omega anyway
    x = (eye(size(x)) + T/2*Somega(omega))*x;
    
%     %Used when calculating dfx
%     syms wx wy wz  T q1 q2 q3 q4
%     w = [wx;wy;wz];
%     q = [q1;q2;q3;q4];
%     dfx = (eye(size(q)) + T/2*Somega(w))*q;
%     dfx = jacobian(dfx, q);
    
    wx = omega(1);
    wy = omega(2);
    wz = omega(3);
    dfx = [        1, 1 - (T*wx)/2, 1 - (T*wy)/2, 1 - (T*wz)/2
            (T*wx)/2,            0,     (T*wz)/2,    -(T*wy)/2
            (T*wy)/2,    -(T*wz)/2,            0,     (T*wx)/2
            (T*wz)/2,     (T*wy)/2,    -(T*wx)/2,            0];
    
    
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
