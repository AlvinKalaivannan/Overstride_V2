% Entry point for the foot-axis A/B, run detached.
%
% Pass A reproduces the pipeline's original behaviour (flag off) and exists as a
% REGRESSION TEST -- its output must be bit-identical to the pre-patch batch,
% because every existing result in this repository came through that path.
% Pass B measures the foot axis from each subject's own metatarsal landmarks.

addpath('scripts/matlab');

fukuchi_foot_ab('data/derived/matlab_code', 'data/fukuchi/ric_format', ...
                'data/fukuchi/ric_format/curves_footA', ...
                'data/fukuchi/ric_format/manifest_footA.csv', 0);

fukuchi_foot_ab('data/derived/matlab_code', 'data/fukuchi/ric_format', ...
                'data/fukuchi/ric_format/curves_footB', ...
                'data/fukuchi/ric_format/manifest_footB.csv', 1);
