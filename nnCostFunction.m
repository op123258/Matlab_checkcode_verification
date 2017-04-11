function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):(((hidden_layer_size * (input_layer_size + 1)))+hidden_layer_size*(hidden_layer_size+1))), ...
                 hidden_layer_size, (hidden_layer_size + 1));
             
Theta3 = reshape(nn_params((((1 + (hidden_layer_size * (input_layer_size + 1)))+hidden_layer_size*(hidden_layer_size+1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         

X = [ones(m, 1), X];
A2 = [ones(m, 1), sigmoid(X*Theta1')];
A3 = [ones(m, 1), sigmoid(A2*Theta2')];
A4 = sigmoid(A3*Theta3');

temp_y = zeros(m, num_labels);
for c = 1:num_labels
    temp_y(:,c) = (y == c);
end
J = -1*sum(sum((temp_y.*log(A4)) + ((1-temp_y).*log((1-A4)))))/m+lambda*(sum(sum((Theta1(:, 2:end)).^2))+sum(sum((Theta2(:, 2:end)).^2))+...
    sum(sum((Theta3(:, 2:end)).^2)))/(2*m);

d4 = A4 - temp_y;
d3 = d4*Theta3.*A3.*(1-A3);
d3 = d3(:, 2:end);
d2 = d3*Theta2.*A2.*(1-A2);
d2 = d2(:, 2:end);


Theta1_grad = 1/m*(d2'*X +lambda*[zeros(hidden_layer_size, 1), Theta1(:, 2:end)]);
Theta2_grad = 1/m*(d3'*A2 +lambda*[zeros(hidden_layer_size, 1), Theta2(:, 2:end)]);
Theta3_grad = 1/m*(d4'*A3 +lambda*[zeros(num_labels, 1), Theta3(:, 2:end)]);


grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end