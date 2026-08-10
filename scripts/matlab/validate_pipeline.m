function validate_pipeline(codeFolder, listFile, outFile, forcedLabel)
%VALIDATE_PIPELINE Run the bundled RIC pipeline on a few sessions and dump the
% outputs so Python can regression-test them against the dv_r already stored in
% the archive.
%
% The archive ships its own regression test: gait_steps returns
% DISCRETE_VARIABLES, and the JSONs store dv_r produced by this same pipeline.
% If they agree, the 101-point curves emitted alongside them are trustworthy.
%
% INPUTS
%   codeFolder - folder containing gait_kinematics.m / gait_steps.m
%   listFile   - text file, one absolute JSON path per line
%   outFile    - .mat path to write results to
%
% Called headless as:
%   matlab -batch "addpath('scripts/matlab'); validate_pipeline(code,list,out)"

addpath(codeFolder);

% The vendored pipeline takes the walk/run label from here rather than from the
% LDA in gaitClass.mat, which MATLAB R2026a cannot deserialize. Every session in
% this list came from run_data_meta.csv and we pass out.running, so 'run' is
% known ground truth, not a guess. See scripts/vendor_matlab_code.py.
if nargin < 4 || isempty(forcedLabel)
    forcedLabel = 'run';
end
global RIC_FORCED_LABEL %#ok<GVMIS>
RIC_FORCED_LABEL = forcedLabel;

fid = fopen(listFile, 'r');
paths = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
paths = paths{1};

results = struct('path', {}, 'ok', {}, 'err', {}, 'dv', {}, 'label', {}, ...
                 'speed', {}, 'nsteps_L', {}, 'nsteps_R', {}, ...
                 'ang_size', {}, 'ang_fields', {}, 'secs', {}, ...
                 'eventsflag_mean', {});

for i = 1:numel(paths)
    jsonFile = strtrim(paths{i});
    rec = struct('path', jsonFile, 'ok', false, 'err', '', 'dv', [], ...
                 'label', '', 'speed', NaN, 'nsteps_L', NaN, 'nsteps_R', NaN, ...
                 'ang_size', [], 'ang_fields', {{}}, 'secs', NaN, ...
                 'eventsflag_mean', NaN);
    t0 = tic;
    try
        raw = fileread(jsonFile);
        out = jsondecode(raw);

        % IMPORTANT (from processing_code_example.m line 55): writing to JSON
        % does not faithfully recreate the structure of out.joints / out.neutral
        % as stored in the original .MAT. They must be transposed before input.
        fn = fieldnames(out.joints);
        for j = 1:numel(fn)
            out.joints.(fn{j}) = transpose(out.joints.(fn{j}));
        end
        fn = fieldnames(out.neutral);
        for j = 1:numel(fn)
            out.neutral.(fn{j}) = transpose(out.neutral.(fn{j}));
        end

        if ~isfield(out, 'dv_r') || isempty(out.running)
            error('no running data in session');
        end

        [r_ang, r_vel] = gait_kinematics(out.joints, out.neutral, ...
                                         out.running, out.hz_r, 0);
        [r_norm_ang, ~, ~, ~, DV, spd, evflag, lbl] = gait_steps( ...
            out.neutral, out.running, r_ang, r_vel, out.hz_r, 0);

        rec.ok = true;
        rec.dv = DV;
        rec.label = char(lbl);
        rec.speed = spd;
        rec.ang_fields = fieldnames(r_norm_ang);
        sz = size(r_norm_ang.L_ankle);
        rec.ang_size = sz;
        rec.nsteps_L = sz(2);
        rec.nsteps_R = size(r_norm_ang.R_ankle, 2);
        rec.eventsflag_mean = mean(evflag(:));
    catch ME
        rec.err = sprintf('%s: %s', ME.identifier, ME.message);
    end
    rec.secs = toc(t0);
    results(end+1) = rec; %#ok<AGROW>
    fprintf('%d/%d  ok=%d  %.1fs  %s\n', i, numel(paths), rec.ok, rec.secs, ...
            jsonFile);
end

save(outFile, 'results', '-v7');
fprintf('wrote %s\n', outFile);
end
