% run_smoke_test.m
% Run from the repository root in MATLAB.
% This script checks that required toolboxes and functions are available.


addpath('matlab');

% 1. Check main runner
assert(exist('run_lhrf', 'file') == 2, ...
    'MATLAB runner "run_lhrf.m" is missing. Ensure the "matlab/" folder is on the path.');

% 2. Check Computer Vision Toolbox (SURF, BoVW, etc.)
assert(exist('detectSURFFeatures', 'file') ~= 0, ...
    'Computer Vision Toolbox is required. Function "detectSURFFeatures" not found.');
assert(exist('extractFeatures', 'file') ~= 0, ...
    'Computer Vision Toolbox is required. Function "extractFeatures" not found.');

% 3. Check Deep Learning Toolbox (SAE, FC-ResNet, Triplets)
assert(exist('trainNetwork', 'file') ~= 0, ...
    'Deep Learning Toolbox is required. Function "trainNetwork" not found.');
assert(exist('dlarray', 'file') ~= 0, ...
    'Deep Learning Toolbox is required. Function "dlarray" not found.');
assert(exist('adamupdate', 'file') ~= 0, ...
    'Deep Learning Toolbox is required. Function "adamupdate" not found.');

% 4. Check Statistics and Machine Learning Toolbox (K-means, HC, Metrics)
assert(exist('kmeans', 'file') ~= 0, ...
    'Statistics and Machine Learning Toolbox is required. Function "kmeans" not found.');
assert(exist('linkage', 'file') ~= 0, ...
    'Statistics and Machine Learning Toolbox is required. Function "linkage" not found.');
assert(exist('perfcurve', 'file') ~= 0, ...
    'Statistics and Machine Learning Toolbox is required. Function "perfcurve" not found.');

disp('MATLAB dependency smoke test passed: all required toolboxes and functions are available.');
