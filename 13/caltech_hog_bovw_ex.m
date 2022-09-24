%--------------------------------------------
%some configurations
addpath('mex'); % add path to external lib
opts.datasetpath = '101_ObjectCategories.rar';
opts.codebookfts = 100000; %number of features for codebook learning
opts.codebooksiz = 200; %number of codebook items

%--------------------------------------------
%creat dataset for experiment
classes = dir(opts.datasetpath); % list up classes
classes(1:3) = []; % remove empty and background classes
num = 1;
for i = 1:length(classes)
    imlist = dir(fullfile(opts.datasetpath, classes(i).name, '\*.jpg')); % list images in the ith class
    for j = 1:25 % indeces of images for training
        dataset(num).classname = classes(i).name;
        dataset(num).imagename = imlist(j).name;
        dataset(num).classID   = i;
        dataset(num).set = 1; %training set
        num = num + 1;
    end
    for j = 26:30 % indeces of images for test
        dataset(num).classname = classes(i).name;
        dataset(num).imagename = imlist(j).name;
        dataset(num).classID   = i;
        dataset(num).set = 2; %test set
        num = num + 1;
    end
end
clear classes num i j imlist;

%--------------------------------------------
%compute local features
%---------------------
%case 1: compute hog as local features
for i = 1:length(dataset)
    im_i = imread(fullfile(opts.datasetpath, dataset(i).classname, dataset(i).imagename));
    ft_i = vl_hog(single(im_i), 8, 'variant', 'dalaltriggs'); % use hog in this sample, it is free to replace it with sift or others
    ft_i = reshape(ft_i, size(ft_i,1)*size(ft_i,2), size(ft_i,3))'; % we do not care the location of features, so reshape the feature map to 36-d vectors   
    dataset(i).locFeat = ft_i;
    fprintf('Compute Local Feature: %d of %d\n', i, length(dataset));
end
clear i im_i ft_i;
%---------------------
%case 2: compute sift as local features
% for i = 1:length(dataset)
%     im_i = imread(fullfile(opts.datasetpath, dataset(i).classname, dataset(i).imagename));
%     if size(im_i,3)==3
%         im_i = rgb2gray(im_i);
%     end
%     im_i = single((im_i));
%     [~, ft_i] = vl_sift(im_i);   
%     dataset(i).locFeat = single(ft_i);
%     fprintf('Compute Local Feature: %d of %d\n', i, length(dataset));
% end
% clear i im_i ft_i;

%--------------------------------------------
%learn codebook using training data
trlist = find(horzcat(dataset(:).set)==1); % find train images
trfeat = horzcat(dataset(trlist).locFeat); % get features from all training images
trfeat = trfeat(:,randperm(length(trfeat), opts.codebookfts));% random sample local features for codebook learning
codebook = vl_kmeans(trfeat, opts.codebooksiz);%learn codebook using kmeans
clear trlist trfeat;

%--------------------------------------------
% compute histogram of codebook items
for i = 1:length(dataset)
    hist = mv_hist(dataset(i).locFeat, codebook); % compute occurances of codebook items
    hist = hist/sum(hist); % l1-norm normalization
    dataset(i).bowFeat = hist;
    fprintf('Compute BoW Feature: %d of %d\n', i, length(dataset));
end
clear i j dist hist indx;
%save dataset dataset -v7.3;

%--------------------------------------------
% classification and evaluation
trlist = find(horzcat(dataset(:).set)==1);
telist = find(horzcat(dataset(:).set)==2);
tr_x = horzcat(dataset(trlist).bowFeat);
tr_y = horzcat(dataset(trlist).classID);
te_x = horzcat(dataset(telist).bowFeat);
te_y = horzcat(dataset(telist).classID);
clear trlist telist;

predicts = mv_1nn(tr_x, tr_y, te_x);
accuracy = sum(predicts==te_y)/length(te_y);
fprintf('The recognition accuracy is %s%% !\n', num2str(accuracy*100));

save dataset dataset -v7.3;
save codebook codebook -v7.3;