function G = mysigmoid2( U, V )
%MYSIGMOID2 Sigmoid kernel function with slope gamma and intercept c
gamma = 0.5;
c = -1;
G = tanh(gamma*U*V' + c);
end

