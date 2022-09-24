% SIFT features
% Tian Xiaoyang

Input = imread('/Users/wcwe/Desktop/101_ObjectCategories/Faces/image_0003.jpg');
F = vl_sift(Input);
points = detectSIFTFeatures(F);
imshow(F);
hold on;
plot(points.selectStrongest(10))