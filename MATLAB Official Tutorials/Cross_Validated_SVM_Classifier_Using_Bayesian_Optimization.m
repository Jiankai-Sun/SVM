% Generate the 10 base points for each class.
rng default
grnpop = mvnrnd([1, 0], eye(2), 10);
redpop = mvnrnd([0, 1], eye(2), 10);
% View the base points
plot(grnpop(:,1), grnpop(:,2), 'go')
hold on
plot(redpop(:,1), redpop(:,2), 'ro')
hold off
% Generate the 100 data points of each class.
redpts = zeros(100, 2); grnpts = redpts;
for i=1:100
    grnpts(i, :) = mvnrnd(grnpop(randi(10),:), eye(2) * 0.02);
    redpts(i, :) = mvnrnd(redpop(randi(10),:), eye(2) * 0.02);
end
% View the data points
figure
plot(grnpts(:,1), grnpts(:,2),'go')
hold on
plot(redpts(:,1), redpts(:,2),'ro')
hold off
% Put the data into one matrix, and make a vector `grp` that labels the class of each point.
cdata = [grnpts;redpts]
grp = ones(200, 1);
% Green label 1, red label -1
grp(101:200) = -1;
% Set up a partition for cross-validation.
c = cvpartition(200, 'KFold', 10);
% Prepare Variables for Bayesian Optimization
sigma = optimizableVariable('sigma', [1e-5, 1e5], 'Transform', 'log');
box = optimizableVariable('box', [1e-5, 1e5], 'Transform', 'log');
% Objective Function
minfn = @(z)kfoldLoss(fitcsvm(cdata, grp, 'CVPartition', c, ...
                      'KernelFunction', 'rbf', 'BoxConstraint', z.box, ...
                      'KernelScale', z.sigma));
% Optimize Classifier
results = bayesopt(minfn, [sigma, box], 'IsObjectiveDeterministic', true, ...
                   'AcquisitionFunctionName', 'expected-improvement-plus')
% Use the results to train a new, optimized SVM classifier.
z(1) = results.XAtMinObjective.sigma;
z(2) = results.XAtMinObjective.box;
SVMModel = fitcsvm(cdata, grp, 'KernelFunction', 'rbf', ...
                   'KernelScale', z(1), 'BoxConstraint', z(2));
% Plot the classification boundaries. 
% To visualize the support vector classifier, predict scores over a grid.
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(cdata(:, 1)):d:max(cdata(:, 1)), ...
                            min(cdata(:, 2)):d:max(cdata(:, 2)));
xGrid = [x1Grid(:), x2Grid(:)];
[~, scores] = predict(SVMModel, xGrid);

h = nan(3, 1);  % Preallocation
figure;
h(1:2) = gscatter(cdata(:, 1), cdata(:, 2), grp, 'rg', '+*');
hold on
h(3) = plot(cdata(SVMModel.IsSupportVector, 1), ...
            cdata(SVMModel.IsSupportVector, 2), 'ko');
contour(x1Grid, x2Grid, reshape(scores(:, 2), size(x1Grid)), [0 0], 'k');
legend(h, {'-1', '+1', 'Support Vectors'}, 'Location', 'Southeast');
axis equal
hold off
% Evaluate Accuracy on New Data
grnobj = gmdistribution(grnpop, .2 * eye(2));
redobj = gmdistribution(redpop, .2 * eye(2));

newData = random(grnobj, 10);
newData = [newData; random(redobj, 10)];
grpData = ones(20, 1);
grpData(11:20) = -1; % red = -1

v = predict(SVMModel, newData);

g = nan(7,1);
figure;
h(1:2) = gscatter(cdata(:, 1), cdata(:, 2), grp, 'rg', '+*');
hold on
h(3:4) = gscatter(newData(:, 1), newData(:, 2), v, 'mc', '**');
h(5) = plot(cdata(SVMModel.IsSupportVector, 1), ...
            cdata(SVMModel.IsSupportVector, 2), 'ko');
contour(x1Grid, x2Grid, reshape(scores(:, 2), size(x1Grid)), [0 0], 'k');
legend(h(1:5), {'-1 (training)', '+1 (training)', '-1 (classified)', ...
                '+1 (classified)', 'Support Vectors'}, 'Location', 'Southeast');
axis equal
hold off

% See which new data points are correctly classified. 
% Circle the correctly classified points in red, 
% and the incorrectly classified points in black.
mydiff = (v == grpData); % Classified correctly
figure;
h(1:2) = gscatter(cdata(:, 1), cdata(:, 2), grp, 'rg', '+*');
hold on
h(3:4) = gscatter(newData(:, 1), newData(:, 2), v, 'mc', '**');
h(5) = plot(cdata(SVMModel.IsSupportVector, 1), ...
            cdata(SVMModel.IsSupportVector, 2), 'ko');
contour(x1Grid, x2Grid, reshape(scores(:, 2), size(x1Grid)), [0 0], 'k');

for ii = mydiff % Plot red squares around correct pts
    h(6) = plot(newData(ii, 1), newData(ii, 2), 'rs', 'MarkerSize', 12);
end

for ii = not(mydiff) % Plot black squares around incorrect pts
    h(7) = plot(newData(ii, 1), newData(ii, 2), 'ks', 'MarkerSize', 12);
end

legend(h, {'-1 (training)', '+1 (training)' ,'-1 (classified)', ...
           '+1 (classified)', 'Support Vectors', 'Correctly Classified', ...
           'Misclassified'}, 'Location', 'Southeast');
hold off

    



