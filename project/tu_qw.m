function [x,P] = tu_qw(x, P, omega, T, Rw)
    
    F = (eye(size(x,1)) + T/2*Somega(omega));
        
    
    x = F*x;
    %rad2deg(q2euler(x))
    
    %Used when calculating dgv and dgq analytically
%      syms v1 v2 v3  T q1 q2 q3 q4
%      v = [v1;v2;v3];
%      q = [q1;q2;q3;q4];
%      g = T/2*Sq(q)*v;
%      dgv = jacobian(g, v);
%      dgq = jacobian(g, q);
    
    dgq = zeros(4);
    q1 = x(1);
    q2 = x(2);
    q3 = x(3);
    q4 = x(4); 
    dgv =   [-(T*q2)/2, -(T*q3)/2, -(T*q4)/2
            (T*q1)/2, -(T*q4)/2,  (T*q3)/2
            (T*q4)/2,  (T*q1)/2, -(T*q2)/2
            -(T*q3)/2,  (T*q2)/2,  (T*q1)/2];
     
    
    P = F*P*F.' + dgv*Rw*dgv.' + dgq*P*dgq.'; 
    [x, P] = mu_normalizeQ(x, P);
end
