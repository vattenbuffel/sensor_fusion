function [x, P, outlier] = mu_g(x, P, yacc, Ra, g0)
    
    error_margin = 0.2;
    outlier = true;
    %fprintf("norm(yacc) == %f\n", norm(yacc))
    if  norm(yacc) < 9.81 + error_margin && norm(yacc) > 9.81 - error_margin
        outlier = false;
        h = Qq(x).'*g0;
        % Used when calculating h_der analytically
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
    
    if outlier
       %fprintf("found outlier\n") 
    end
    
    [x, P] = mu_normalizeQ(x, P);
end