% Load Fisher's iris data set. 
% Train the classifier using the petal lengths and widths, 
% and remove the virginica species from the data.
load fisheriris
classKeep = ~strcmp(species, 'virginica');
X = meas(classKeep, 3:4);
y = species(classKeep);
% Train an SVM classifier using the data. 
% It is good practice to specify the order of the classes.
SVMModel = fitcsvm(X, y, 'ClassNames', {'setosa', 'versicolor'});
% Estimate the optimal score transformation function.
rng(1); % For reproducibility
[SVMModel, ScoreParameters] = fitPosterior(SVMModel);
ScoreParameters
% Define a grid of values in the observed predictor space.
% Predict the posterior probabilities for each instance in the grid.
xMax = max(X);
xMin = min(X);
d = 0.01;
[x1Grid, x2Grid] = meshgrid(xMin(1):d:xMax(1), xMin(2):d:xMax(2));
[~, PosteriorRegion] = predict(SVMModel, [x1Grid(:), x2Grid(:)]);
% Plot the positive class posterior probability region and the training data.
figure;
contourf(x1Grid, x2Grid, ...
        reshape(PosteriorRegion(:, 2), size(x1Grid, 1), size(x1Grid, 2)));
h = colorbar;
h.Label.String = 'P({\it{versicolor}})';
h.YLabel.FontSize = 16;
caxis([0 1]);
colormap jet;

hold on
gscatter(X(:,1), X(:, 2), y, 'mc', '.x', [15, 10]);
sv = X(SVMModel.IsSupportVector, :);
plot(sv(:, 1), sv(:, 2), 'yo','MarkerSize', 15, 'LineWidth', 2);
axis tight
hold off

