function [x, P, L, outlier] = mu_m(x, P, mag, m0, Rm, L, alpha)
    h = Qq(x).'*m0;
    % Used when calculating h_der analytically
%     syms q1 q2 q3 q4
%     q = [q1;q2;q3;q4];
%     h_der = Qq(q).'*m0;
%     h_der = jacobian(h_der, q);
    error_margin = 0.1;
    L = (1-alpha)*L + alpha*norm(mag);
    outlier = true;
    
    
    if  norm(mag) < L + error_margin && norm(mag) > L - error_margin
        outlier = false;
        q1 = x(1);
        q2 = x(2);
        q3 = x(3);
        q4 = x(4);
        h_der =  [   (44091*q3)/625 + (1626707222280641*q4)/35184372088832, (1626707222280641*q3)/35184372088832 - (44091*q4)/625, (44091*q1)/625 + (1626707222280641*q2)/35184372088832, (1626707222280641*q1)/35184372088832 - (44091*q2)/625
                     (1626707222280641*q1)/17592186044416 - (44091*q2)/625,                                       -(44091*q1)/625, (1626707222280641*q3)/17592186044416 - (44091*q4)/625,                                       -(44091*q3)/625
                   - (88182*q1)/625 - (1626707222280641*q2)/35184372088832,                 -(1626707222280641*q1)/35184372088832,                  (1626707222280641*q4)/35184372088832, (1626707222280641*q3)/35184372088832 - (88182*q4)/625];


        S = h_der*P*h_der.' + Rm;
        K = P*h_der.'*S^-1;
        x = x + K*(mag-h);
        P = P - K*S*K.';
    end
    [x, P] = mu_normalizeQ(x, P);

end