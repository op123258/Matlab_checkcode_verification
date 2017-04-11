input_layer_size  = 260 ;  
hidden_layer_size = 200;   
num_labels = 36;            
                          
load('sample.mat');
% load('THETA.mat');
y = Y;

m = size(X, 1);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ];

% initial_nn_params = [Theta1(:); Theta2(:); Theta3(:)];


%% =================== 训练模型 ===================

%最大迭代次数100
options = optimset('MaxIter', 100);

lambda = 0.6;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):...
                 (((hidden_layer_size * (input_layer_size + 1)))+hidden_layer_size*(hidden_layer_size+1))), ...
                 hidden_layer_size, (hidden_layer_size + 1));
             
Theta3 = reshape(nn_params((((1 + (hidden_layer_size * (input_layer_size + 1)))+hidden_layer_size*(hidden_layer_size+1))):end), ...
                 num_labels, (hidden_layer_size + 1));



%% ================= 将训练集代入模型，检验模型 =================


pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\n训练所得 模型 与 训练样本 的契合度（精度）: %f\n', mean(double(pred == y)) * 100);
%save THETA Theta1 Theta2 Theta3




