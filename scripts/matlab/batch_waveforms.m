function batch_waveforms(codeFolder, listFile, outDir, manifestFile)
%BATCH_WAVEFORMS Generate 101-point stance-phase curves for every session.
%
% The archive stores raw markers, not waveforms. This runs the bundled RIC
% pipeline (gait_kinematics -> gait_steps) over a list of sessions and saves the
% normalized angle and velocity curves that phases 1C-5 need.
%
% WALK/RUN LABEL RECOVERY
% -----------------------
% gaitClass.mat holds an LDA that MATLAB R2026a cannot deserialize, so the
% vendored pipeline takes the label from RIC_FORCED_LABEL instead (see
% scripts/vendor_matlab_code.py). We do not guess it: gait_steps also returns
% DISCRETE_VARIABLES, and the session JSON already stores dv_r from the original
% run. So each session is processed as 'run', its DISCRETE_VARIABLES compared
% against the stored dv_r, and if they disagree it is reprocessed as 'walk'.
% Whichever reproduces the archive is the label the original classifier chose.
%
% Verified: session 100520/20120524T131220 is filed under running but only
% reproduces as 'walk' (40/40 live variables vs 4/40). Sessions where NEITHER
% label reproduces are recorded as unresolved and must be excluded downstream.
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

mf = fopen(manifestFile, 'w');
fprintf(mf, ['json,sub_id,session,label_used,n_live,n_agree,resolved,' ...
             'speed,nsteps_L,nsteps_R,eventsflag_mean,secs,err\n']);

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
    t0 = tic; errMsg = ''; resolved = 0; bestLabel = ''; bestAgree = -1;
    bestLive = 0; spd = NaN; nsL = NaN; nsR = NaN; evm = NaN; saved = false;

    try
        raw = fileread(jsonFile);
        out = jsondecode(raw);

        % IMPORTANT (processing_code_example.m:55): JSON does not faithfully
        % recreate out.joints / out.neutral. Transpose before input.
        fn = fieldnames(out.joints);
        for j = 1:numel(fn); out.joints.(fn{j}) = transpose(out.joints.(fn{j})); end
        fn = fieldnames(out.neutral);
        for j = 1:numel(fn); out.neutral.(fn{j}) = transpose(out.neutral.(fn{j})); end

        if ~isfield(out, 'dv_r') || isempty(out.running)
            error('no running data in session');
        end

        % Stored dv_r, in field order, as an (76 x 2) reference.
        dvFields = fieldnames(out.dv_r.left);
        stored = nan(numel(dvFields), 2);
        for j = 1:numel(dvFields)
            lv = out.dv_r.left.(dvFields{j});
            rv = out.dv_r.right.(dvFields{j});
            if isnumeric(lv) && isscalar(lv); stored(j,1) = double(lv); end
            if isnumeric(rv) && isscalar(rv); stored(j,2) = double(rv); end
        end
        liveMask = ~( (stored(:,1)==0 | isnan(stored(:,1))) & ...
                      (stored(:,2)==0 | isnan(stored(:,2))) );

        [r_ang, r_vel] = gait_kinematics(out.joints, out.neutral, ...
                                         out.running, out.hz_r, 0);

        for lab = {'run', 'walk'}
            RIC_FORCED_LABEL = lab{1};
            [na, nv, ~, ~, DV, spd_i, evflag, ~] = gait_steps( ...
                out.neutral, out.running, r_ang, r_vel, out.hz_r, 0);

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
            if nAgree == nLive; break; end   % exact -- no need to try 'walk'
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
    fprintf(mf, '%s,%s,%s,%s,%d,%d,%d,%.6f,%d,%d,%.4f,%.2f,%s\n', ...
            jsonFile, subId, sessName, bestLabel, bestLive, bestAgree, ...
            resolved, spd, nsL, nsR, evm, secs, errMsg);

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
