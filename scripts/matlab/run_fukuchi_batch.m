% Entry point for the detached Fukuchi batch.
%
% A file rather than a -batch string: passing the call through Start-Process
% mangles embedded quotes, and MATLAB then exits instantly writing nothing to
% either stream, which is indistinguishable from still starting up.
%
% Resumable -- fukuchi_batch skips trials already in the manifest.

addpath('scripts/matlab');
fukuchi_batch('data/derived/matlab_code', ...
              'data/fukuchi/ric_format', ...
              'data/fukuchi/ric_format/curves', ...
              'data/fukuchi/ric_format/batch_manifest.csv');
