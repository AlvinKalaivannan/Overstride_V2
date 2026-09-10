% Entry point for the detached walking batch.
%
% Exists so the job can be launched as `matlab -batch run_walk_batch` with no
% quoting: passing the whole call as a -batch string through Start-Process
% mangled the embedded single quotes and MATLAB exited immediately, writing
% nothing to either stream. A file has no such failure mode.
%
% Resumable: batch_waveforms_walk skips sessions already in the manifest, so
% re-running this after any interruption picks up where it stopped.

addpath('scripts/matlab');
batch_waveforms_walk('data/derived/matlab_code', ...
                     'data/derived/walk_list.txt', ...
                     'data/derived/walk_steps', ...
                     'data/derived/walk_manifest.csv');
