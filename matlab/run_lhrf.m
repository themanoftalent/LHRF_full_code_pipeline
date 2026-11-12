function results = run_lhrf(manifestPath, outputDir, useHC)
% LHRF MATLAB implementation (Experimental Revision).
% MATLAB R2022b; Computer Vision, Deep Learning, Statistics toolboxes.
% Manifest CSV: path,label,group.

if nargin < 3, useHC = false; end
assert(nargin >= 2, 'Provide manifestPath and outputDir');
assert(isfile(manifestPath), 'Manifest CSV not found');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
rng(42, 'twister');

T = readtable(manifestPath, 'TextType', 'string');
assert(all(ismember({'path', 'label', 'group'}, T.Properties.VariableNames)));
assert(all(isfile(T.path)), 'Missing image files');

% Force explicit categorical ordering (0 = negative, 1 = positive)
labels = categorical(T.label, [0 1]);
groups = string(T.group);

assert(numel(unique(labels)) == 2, 'This runner requires two classes');
assert(all(~ismissing(groups) & strlength(groups) > 0), 'Group IDs required');
assert(all(~ismissing(labels)), 'Missing labels');
assert(numel(unique(T.path)) == height(T), 'Duplicate image paths in manifest');
assert(numel(unique(groups)) >= 10, 'Five-fold grouped evaluation requires at least ten groups');

for g = unique(groups)'
    assert(numel(unique(labels(groups == g))) == 1, 'Conflicting labels within a group');
end

ug = unique(groups);
assert(numel(ug) >= 5, 'At least five independent groups required');

% Group-level folds, deterministic and disjoint
foldId = zeros(height(T), 1);
for c = categories(labels)'
    ids = find(arrayfun(@(g) all(labels(groups == g) == c), ug));
    if isempty(ids), continue; end
    assert(numel(ids) >= 5, 'Each class requires at least five independent groups for five folds');
    ids = ids(randperm(numel(ids)));
    for j = 1:numel(ids), foldId(groups == ug(ids(j))) = mod(j-1, 5) + 1; end
end
assert(all(foldId > 0), 'Every group must have a single class and a fold');
assert(all(arrayfun(@(g) numel(unique(foldId(groups == g))) == 1, ug)));

writetable(table(T.path, T.label, T.group, foldId, 'VariableNames', {'path', 'label', 'group', 'fold'}), ...
    fullfile(outputDir, 'fold_manifest.csv'));

rows = []; predictions = table();
for fold = 1:5
    tr = foldId ~= fold; te = foldId == fold;
    assert(any(tr) && any(te));
    assert(numel(unique(labels(tr))) == 2 && numel(unique(labels(te))) == 2, ...
        'Every train/test fold must contain both classes');
    assert(isempty(intersect(unique(groups(tr)), unique(groups(te)))), ...
        'Group leakage between training and test');

    [trainDesc, testDesc] = descriptors(T.path(tr), T.path(te));
    
    % Filter out empty descriptor sets
    trainDesc = trainDesc(~cellfun('isempty', trainDesc));
    testDesc  = testDesc(~cellfun('isempty', testDesc));
    
    pool = vertcat(trainDesc{:});
    assert(size(pool, 1) >= 150, 'Not enough SURF features to build 150-word vocabulary');
    
    if size(pool, 1) > 50000
        pool = pool(randperm(size(pool, 1), 50000), :);
    end
    
    [~, vocab] = kmeans(double(pool), 150, 'Replicates', 2, 'MaxIter', 150, 'EmptyAction', 'singleton');
    Xtr = encode(trainDesc, vocab);
    Xte = encode(testDesc, vocab);
    
    mu = mean(Xtr, 1); sd = std(Xtr, 0, 1); sd(sd < 1e-8) = 1;
    Xtr = (Xtr - mu) ./ sd;
    Xte = (Xte - mu) ./ sd;

    % SAE 150 -> 100 -> 50 -> 100 -> 150. Training-only fit.
    layers = [
        featureInputLayer(150, 'Normalization', 'none')
        fullyConnectedLayer(100) reluLayer
        fullyConnectedLayer(50, 'Name', 'latent') reluLayer('Name', 'latent_relu')
        fullyConnectedLayer(100) reluLayer
        fullyConnectedLayer(150) regressionLayer];
    
    opts = trainingOptions('adam', 'InitialLearnRate', 0.001, 'MaxEpochs', 100, ...
        'MiniBatchSize', 32, 'L2Regularization', 0.001, 'Verbose', false);
    
    ae = trainNetwork(Xtr, Xtr, layers, opts);
    
    Ztr = activations(ae, Xtr, 'latent_relu', 'OutputAs', 'rows');
    Zte = activations(ae, Xte, 'latent_relu', 'OutputAs', 'rows');

    % Trainable metric projection, optimized with batch-hard triplet loss
    [Etr, Ete] = tripletProjection(Ztr, Zte, labels(tr), 42 + fold);

    if useHC
        tree = linkage(Etr, 'ward', 'euclidean');
        clusterId = cluster(tree, 'maxclust', 2);
        centers = [mean(Etr(clusterId == 1, :), 1); mean(Etr(clusterId == 2, :), 1)];
        [~, testId] = min(pdist2(Ete, centers), [], 2);
        
        Etr = [Etr double(clusterId == 1) double(clusterId == 2)];
        Ete = [Ete double(testId == 1) double(testId == 2)];
    end

    % Residual MLP with learned projection shortcuts
    net = buildResidual(size(Etr, 2));
    opt = trainingOptions('adam', 'InitialLearnRate', 0.001, 'MaxEpochs', 100, ...
        'MiniBatchSize', 32, 'Shuffle', 'every-epoch', 'Verbose', false);
    
    net = trainNetwork(Etr, labels(tr), net, opt);

    % Predict and compute metrics
    [pred, score] = classify(net, Ete);
    
    cls = categories(labels);
    pos = cls{2};
    classOrder = string(net.Layers(end).Classes);
    posIndex = find(classOrder == string(pos), 1);
    assert(~isempty(posIndex), 'Positive class missing from classifier outputs');
    
    % Score matrix for positive class
    posScores = score(:, posIndex);
    
    [~, ~, ~, auc] = perfcurve(labels(te), posScores, pos);
    cm = confusionmat(labels(te), pred, 'Order', categorical(cls));
    acc = mean(pred == labels(te));
    
    rows = [rows; fold sum(te) acc auc]; %#ok<AGROW>
    
    Q = table(T.path(te), string(labels(te)), string(pred), posScores, repmat(fold, sum(te), 1), ...
        'VariableNames', {'path', 'true_label', 'predicted_label', 'positive_score', 'fold'});
    predictions = [predictions; Q]; %#ok<AGROW>
    
    writematrix(cm, fullfile(outputDir, sprintf('confusion_fold_%d.csv', fold)));
    save(fullfile(outputDir, sprintf('model_fold_%d.mat', fold)), ...
        'net', 'ae', 'vocab', 'mu', 'sd', '-v7.3');
end

results = array2table(rows, 'VariableNames', {'fold', 'n', 'accuracy', 'auc'});
writetable(results, fullfile(outputDir, 'fold_metrics.csv'));
writetable(predictions, fullfile(outputDir, 'predictions.csv'));
save(fullfile(outputDir, 'results.mat'), 'results', 'predictions', 'foldId');
end

function [a, b] = descriptors(pa, pb)
a = cell(numel(pa), 1); b = cell(numel(pb), 1);
for i = 1:numel(pa), a{i} = surfOne(pa(i)); end
for i = 1:numel(pb), b{i} = surfOne(pb(i)); end
end

function d = surfOne(p)
I = imread(p);
if size(I, 3) == 3, I = rgb2gray(I); end
I = imresize(I, [224 224]);
pts = detectSURFFeatures(I);
if pts.Count == 0
    d = zeros(0, 64); % Return empty array instead of a single row
    return;
end
[f, ~] = extractFeatures(I, pts, 'Method', 'SURF');
d = double(f);
end

function X = encode(D, C)
X = zeros(numel(D), size(C, 1));
for i = 1:numel(D)
    if isempty(D{i}), continue; end
    [~, ix] = min(pdist2(double(D{i}), C), [], 2);
    X(i, :) = accumarray(ix, 1, [size(C, 1), 1])';
    X(i, :) = X(i, :) / max(1, sum(X(i, :)));
end
end

function [A, B] = tripletProjection(X, Y, labels, seed)
rng(seed);
d = size(X, 2);
W = dlarray(single(eye(d)));
velocity = []; variance = [];
classes = categories(labels);
ids1 = find(labels == classes{1});
ids2 = find(labels == classes{2});

for epoch = 1:50
    for step = 1:max(1, ceil(size(X, 1) / 32))
        ix = [ids1(randi(numel(ids1), 16, 1)); ids2(randi(numel(ids2), 16, 1))];
        [loss, grad] = dlfeval(@tripletGrad, W, single(X(ix, :)), double(labels(ix) == classes{2})); %#ok<ASGLU>
        [W, velocity, variance] = adamupdate(W, grad, velocity, variance, (epoch-1)*ceil(size(X, 1)/32)+step, 0.001);
    end
end
A = double(extractdata(dlarray(single(X)) * W));
B = double(extractdata(dlarray(single(Y)) * W));
end

function [loss, grad] = tripletGrad(W, X, y)
Z = dlarray(X) * W;
n = size(X, 1);
terms = dlarray(zeros(1, 1));
count = 0;

for i = 1:n
    same = find(y == y(i)); same(same == i) = [];
    other = find(y ~= y(i));
    if isempty(same) || isempty(other), continue; end
    
    ds = sum((Z(same, :) - Z(i, :)).^2, 2);
    dn = sum((Z(other, :) - Z(i, :)).^2, 2);
    
    terms = terms + max(0, max(ds) - min(dn) + 0.2);
    count = count + 1;
end

if count > 0
    loss = terms / count;
else
    loss = dlarray(0);
end
grad = dlgradient(loss, W);
end

function graph = buildResidual(d)
graph = layerGraph(featureInputLayer(d, 'Normalization', 'none', 'Name', 'input'));
prev = 'input'; widths = [300 200 100 70]; old = d;

for k = 1:numel(widths)
    w = widths(k); prefix = sprintf('r%d_', k);
    main = [fullyConnectedLayer(w, 'Name', [prefix 'fc1']) reluLayer('Name', [prefix 'relu1'])...
            fullyConnectedLayer(w, 'Name', [prefix 'fc2'])];
    skip = fullyConnectedLayer(w, 'Name', [prefix 'projection']);
    add = additionLayer(2, 'Name', [prefix 'add']); relu = reluLayer('Name', [prefix 'out']);
    
    graph = addLayers(graph, main);
    graph = addLayers(graph, skip);
    graph = addLayers(graph, add);
    graph = addLayers(graph, relu);
    
    graph = connectLayers(graph, prev, [prefix 'fc1']);
    graph = connectLayers(graph, prev, [prefix 'projection']);
    graph = connectLayers(graph, [prefix 'fc2'], [prefix 'add/in1']);
    graph = connectLayers(graph, [prefix 'projection'], [prefix 'add/in2']);
    graph = connectLayers(graph, [prefix 'add'], [prefix 'out']);
    
    prev = [prefix 'out']; old = w; %#ok<NASGU>
end

out = [fullyConnectedLayer(2, 'Name', 'logits') softmaxLayer('Name', 'softmax') classificationLayer('Name', 'class')];
graph = addLayers(graph, out);
graph = connectLayers(graph, prev, 'logits');
end
