%%Mohsen abyari
%%Covid_19_Lung_scan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

path='D:\MATLAB\Matlab tajhiz teb\Covid_19 lung CT scan\CT_Covid19\*.png';
file=dir(path);

    for     i=1:length(file);
         fn=[path(1:end-5),file(i,1).name];
         im= imread(fn); 
         im1= histeq(im);
       if size(im1, 3) > 1
         im2= rgb2gray(im1);
end
    sigma=0.4;%%sigma characterizes the amplitude of edges in Im
    alpha=0.5;%%alpha controls smoothing of details.
         im3= locallapfilt(im2,sigma,alpha);%%im3 to local laplacian filter
         im4=im2double(im3);%%picture to matrix
         im5=medfilt2(im4,[3,3]);
         [cA,cH,cV,cD] = dwt2(im5,'db10');
         
       
              f1=max(im5(:));%max temp
              f2=min(im5(:));%%min temp
              f3=mean(im5(:));%%mean temp
              f4=entropy(im5(:));%%bi nazmi(-1........+1)
              f5=kurtosis(im5(:));%%pulling (keshidegiu)
              f6=skewness(im5(:));%%khamidegi
              f7=std(im5(:));%%stand of temp
              f8=extractLBPFeatures(im5);
          feature(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8];%%%make feature matrix
            
    
    end
%     subplot(2,2,1)
% imagesc(cA)
% colormap gray
% title('Approximation')
% subplot(2,2,2)
% imagesc(cH)
% colormap gray
% title('Horizontal')
% subplot(2,2,3)
% imagesc(cV)
% colormap gray
% title('Vertical')
% subplot(2,2,4)
% imagesc(cD)
% colormap gray
% title('Diagonal')
    
% %%figure
% %imshow(im)
% %title('Covid_19')
% 
% %figure
% imshow(im1)
% title('Covid_19_histeq')
% 
% figure
% imshow(im3)
% title('Covid_19_locallapfilt')
% 
% figure
% imshow(im5)
% title('Covid_19_medfilt')
 input= feature;

 output=[zeros(272,1);ones(272,1)];
 
 x = input';
t = output';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 5;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 5/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
figure, plotconfusion(t,y)
%figure, plotroc(t,y)



