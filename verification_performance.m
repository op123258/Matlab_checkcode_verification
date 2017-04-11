clear;
clc;

fprintf('\n从训练集中随机加载一张验证码，请按下Enter键继续\n');
pause();
path = './TrainingSet/11.jpg';
im=imread(path);
imshow(im)
set (gcf,'Position',[200,300,600,300]);

fprintf('\n将验证码进行滤波，请按下Enter键继续\n');
pause();
im = im(1:20, :);
% new = wiener2(im);
% new = myfliter(im);
% imshow(new);
gausFilter = fspecial('gaussian',[5 5],0.53);
new=imfilter(im,gausFilter,'replicate');
imshow(new)
set (gcf,'Position',[200,300,600,300]);

fprintf('\n将图片进行二值化，请按下Enter键继续\n');
pause();
BW =im2bw(new, 0.4); 
imshow(BW);
set (gcf,'Position',[200,300,600,300]);

fprintf('\n将图片进行切割，请按下Enter键继续\n');
pause();
s1 = BW(:, 5:17);
s2 = BW(:, 18:30);
s3 = BW(:, 31:43);
s4 = BW(:, 44:56);
subplot(1, 4, 1)
imshow(s1);
subplot(1, 4, 2)
imshow(s2);
subplot(1, 4, 3)
imshow(s3);
subplot(1, 4, 4)
imshow(s4);

fprintf('\n将每一小块拉平成一行，然后将4部分构成一个4行的矩阵,这就是数据的预处理，请按下Enter键继续\n');
pause();
[p1, p2 ,p3, p4] = jpg_split(BW);
x = [p1; p2; p3; p4];

fprintf('\n导入训练集用神经网络训练模型，请按下Enter键继续\n');
pause();
NN();

fprintf('\n导入训练好的模型，进行识别，请按下Enter键继续\n');
pause();
load THETA;
word = '0123456789abcdefghijklmnopqrstuvwxyz';

fprintf('\n进行预测，请按下Enter键继续\n');
pause();
y = predict(Theta1, Theta2, Theta3, x);
y = word(y');
fprintf('预测结果为: %s \n', y);
subplot(1, 4, 1)
imshow(s1);
title(['预测结果为', y(1)]);
subplot(1, 4, 2)
imshow(s2);
title(['预测结果为', y(2)]);
subplot(1, 4, 3)
imshow(s3);
title(['预测结果为', y(3)]);
subplot(1, 4, 4)
imshow(s4);
title(['预测结果为', y(4)]);

