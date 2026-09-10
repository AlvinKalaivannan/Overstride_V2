function batch_waveforms_walk(codeFolder, listFile, outDir, manifestFile)
%BATCH_WAVEFORMS_WALK 101-point stance curves for WALKING trials.
%
% The walking twin of batch_waveforms.m. The bundled pipeline supports both
% gait modes symmetrically -- processing_code_example.m lines 78-82 give the
% walking call verbatim -- so the only differences from the running batch are
% which fields are read:
%
%     running                walking
%     -------                -------
%     out.running            out.walking      marker trajectories
%     out.hz_r               out.hz_w         sampling rate (mostly 200 / 120)
%     out.dv_r               out.dv_w         stored discrete variables
%
% VALIDATION IS THE SAME AND IT IS THE POINT
% ------------------------------------------
% gaitClass.mat holds an LDA that MATLAB R2026a cannot deserialize, so the
% label comes from RIC_FORCED_LABEL. We do not guess it. gait_steps also
% returns DISCRETE_VARIABLES, and the session JSON stores dv_w from the
% original run -- so each session is processed under both labels and the one
% that reproduces the archive's own dv_w is the label the original classifier
% chose. 'walk' is tried FIRST here (the running batch tries 'run' first),
% since these sessions are drawn from walk_data_meta.csv.
%
% A session where neither label reproduces dv_w is recorded unresolved and MUST
% be excluded downstream. Reproducing dv_w exactly is what makes these curves
% verified rather than assumed, exactly as dv_r did for running.
%
% INPUTS
%   codeFolder   - vendored, patched pipeline (data/derived/matlab_code)
%   listFile     - text file, one absolute JSON path per line
%   outDir       - directory for per-session .mat curve files
%   manifestFile - CSV describing every session processed

addpath(codeFolder);
if ~exist(outDir, 'dir'); mkdir(outDir); end

global RIC_FORCED_LABEL %#ok<GVMIS>

fid = fopen(listFile, 'r');
paths = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
paths = paths{1};
n = numel(paths);

% RESUMABLE. A full pass is ~60 minutes, and an interrupted run must not throw
% away completed work. Sessions already recorded in the manifest are skipped and
% the manifest is appended to, so this can be restarted freely.
done = containers.Map('KeyType', 'char', 'ValueType', 'logical');
if exist(manifestFile, 'file')
    fidm = fopen(manifestFile, 'r');
    fgetl(fidm);                                   % discard header
    while true
        ln = fgetl(fidm);
        if ~ischar(ln); break; end
        c = strsplit(ln, ',');
        if numel(c) >= 3
            done([c{2} '__' c{3}]) = true;
        end
    end
    fclose(fidm);
    fprintf('resuming: %d sessions already in manifest\n', done.Count);
    mf = fopen(manifestFile, 'a');
else
    mf = fopen(manifestFile, 'w');
    fprintf(mf, ['json,sub_id,session,label_used,n_live,n_agree,resolved,' ...
                 'speed,nsteps_L,nsteps_R,eventsflag_mean,hz,secs,err\n']);
end

ANG = {'L_ankle','L_knee','L_hip','L_foot','L_pelvis', ...
       'R_ankle','R_knee','R_hip','R_foot','R_pelvis'};
VEL = {'L_ankle','L_knee','L_hip','L_pelvis', ...
       'R_ankle','R_knee','R_hip','R_pelvis'};

t_start = tic;
for i = 1:n
    jsonFile = strtrim(paths{i});
    [~, sessName] = fileparts(jsonFile);
    parts = strsplit(strrep(jsonFile, '\', '/'), '/');
    subId = parts{end-1};
    if isKey(done, [subId '__' sessName]); continue; end
    t0 = tic; errMsg = ''; resolved = 0; bestLabel = ''; bestAgree = -1;
    bestLive = 0; spd = NaN; nsL = NaN; nsR = NaN; evm = NaN; hz = NaN;
    saved = false;

    try
        raw = fileread(jsonFile);
        out = jsondecode(raw);

        % IMPORTANT (processing_code_example.m:55): JSON does not faithfully
        % recreate out.joints / out.neutral. Transpose before input.
        fn = fieldnames(out.joints);
        for j = 1:numel(fn); out.joints.(fn{j}) = transpose(out.joints.(fn{j})); end
        fn = fieldnames(out.neutral);
        for j = 1:numel(fn); out.neutral.(fn{j}) = transpose(out.neutral.(fn{j})); end

        if ~isfield(out, 'dv_w') || isempty(out.walking)
            error('no walking data in session');
        end
        hz = double(out.hz_w);

        % Stored dv_w, in field order, as an (nfields x 2) reference.
        dvFields = fieldnames(out.dv_w.left);
        stored = nan(numel(dvFields), 2);
        for j = 1:numel(dvFields)
            lv = out.dv_w.left.(dvFields{j});
            rv = out.dv_w.right.(dvFields{j});
            if isnumeric(lv) && isscalar(lv); stored(j,1) = double(lv); end
            if isnumeric(rv) && isscalar(rv); stored(j,2) = double(rv); end
        end
        liveMask = ~( (stored(:,1)==0 | isnan(stored(:,1))) & ...
                      (stored(:,2)==0 | isnan(stored(:,2))) );

        [w_ang, w_vel] = gait_kinematics(out.joints, out.neutral, ...
                                         out.walking, out.hz_w, 0);

        for lab = {'walk', 'run'}
            RIC_FORCED_LABEL = lab{1};
            [na, nv, ~, ~, DV, spd_i, evflag, ~] = gait_steps( ...
                out.neutral, out.walking, w_ang, w_vel, out.hz_w, 0);

            mine = DV(2:end, 2:3);
            agree = abs(mine - stored) <= (1e-6 + 1e-4 * abs(stored));
            agree(isnan(mine) & isnan(stored)) = true;
            nAgree = sum(all(agree(liveMask, :), 2));
            nLive = sum(liveMask);

            if nAgree > bestAgree
                bestAgree = nAgree; bestLive = nLive; bestLabel = lab{1};
                spd = spd_i; evm = mean(evflag(:));
                nsL = size(na.L_ankle, 2); nsR = size(na.R_ankle, 2);
                curves = struct();
                for k = 1:numel(ANG)
                    curves.(['ang_' ANG{k}]) = single(na.(ANG{k}));
                end
                for k = 1:numel(VEL)
                    curves.(['vel_' VEL{k}]) = single(nv.(VEL{k}));
                end
            end
            if nAgree == nLive; break; end   % exact -- no need to try 'run'
        end
        RIC_FORCED_LABEL = '';

        resolved = double(bestAgree == bestLive && bestLive > 0);
        outFile = fullfile(outDir, sprintf('%s__%s.mat', subId, sessName));
        save(outFile, '-struct', 'curves', '-v7');
        saved = true;
    catch ME
        errMsg = strrep(sprintf('%s: %s', ME.identifier, ME.message), ',', ';');
    end

    secs = toc(t0);
    fprintf(mf, '%s,%s,%s,%s,%d,%d,%d,%.6f,%d,%d,%.4f,%.1f,%.2f,%s\n', ...
            jsonFile, subId, sessName, bestLabel, bestLive, bestAgree, ...
            resolved, spd, nsL, nsR, evm, hz, secs, errMsg);

    % Flush periodically so an interrupt loses at most the row in flight.
    if mod(i, 10) == 0; fclose(mf); mf = fopen(manifestFile, 'a'); end
    if mod(i, 25) == 0 || i == n
        el = toc(t_start);
        fprintf('%d/%d  %.2f s/file  elapsed %.1f min  eta %.1f min\n', ...
                i, n, el/i, el/60, (n-i)*el/i/60);
    end
    if ~saved && isempty(errMsg)
        fprintf('  WARN: %s produced no curves\n', jsonFile);
    end
end

fclose(mf);
fprintf('done: %d sessions, manifest -> %s\n', n, manifestFile);
end
