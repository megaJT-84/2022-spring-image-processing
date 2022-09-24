% extract HOG features
% Tian Xiaoyang

img = imread('/Users/wcwe/Desktop/101_ObjectCategories/euphonium/image_0002.jpg');
hog_result = vl_hog(img);
figure;
imshow(img);
hold on;
plot(hogVisualization);