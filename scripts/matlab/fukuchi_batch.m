function fukuchi_batch(codeFolder, jsonDir, outDir, manifestFile)
%FUKUCHI_BATCH Run the RIC pipeline over every adapted Fukuchi trial.
%
% Phase 10's validation harness. Fukuchi has no stored dv_r, so the two-label
% reconciliation the archive batches use is unavailable and the label is forced
% to 'run' -- which is what these trials are.
%
% THE VALIDATION THIS ENABLES
% ---------------------------
% Fukuchi ran each subject on a treadmill at PRESCRIBED speeds (2.5 / 3.5 / 4.5
% m/s), encoded in the filename. `gait_steps` independently computes speed from
% marker kinematics and knows nothing about the filename. So agreement between
% the two is an external check on the whole adapter -- frame, marker mapping,
% cluster assignment and units together. A rotation error large enough to matter
% cannot leave the speed right.
%
% Resumable: trials already in the manifest are skipped.

addpath(codeFolder);
global RIC_FORCED_LABEL %#ok<GVMIS>

if ~exist(outDir, 'dir'); mkdir(outDir); end

done = containers.Map('KeyType', 'char', 'ValueType', 'logical');
if exist(manifestFile, 'file')
    fid = fopen(manifestFile, 'r');
    fgetl(fid);
    while true
        ln = fgetl(fid);
        if ~ischar(ln); break; end
        c = strsplit(ln, ',');
        if numel(c) >= 1; done(c{1}) = true; end
    end
    fclose(fid);
    fprintf('resuming: %d trials already done\n', done.Count);
    mf = fopen(manifestFile, 'a');
else
    mf = fopen(manifestFile, 'w');
    fprintf(mf, ['trial,subject,speed_nominal,speed_computed,nsteps_L,nsteps_R,' ...
                 'eventsflag,hz,n_frames,secs,err\n']);
end

files = dir(fullfile(jsonDir, 'RBDS*run*.json'));
files = files(~contains({files.name}, '.meta.'));
n = numel(files);
fprintf('%d trials to process\n', n);

ANG = {'L_ankle','L_knee','L_hip','L_foot','L_pelvis', ...
       'R_ankle','R_knee','R_hip','R_foot','R_pelvis'};
VEL = {'L_ankle','L_knee','L_hip','L_pelvis','R_ankle','R_knee','R_hip','R_pelvis'};

t_start = tic;
for i = 1:n
    [~, stem] = fileparts(files(i).name);
    if isKey(done, stem); continue; end

    tok = regexp(stem, '^(RBDS\d+)runT(\d+)$', 'tokens');
    if isempty(tok)
        fprintf('  skip %s (unparsable name)\n', stem); continue;
    end
    subj = tok{1}{1};
    nominal = str2double(tok{1}{2}) / 10;      % T25 -> 2.5 m/s

    t0 = tic; errMsg = ''; spd = NaN; nsL = NaN; nsR = NaN; evm = NaN;
    hz = NaN; nfr = NaN;
    try
        out = jsondecode(fileread(fullfile(jsonDir, files(i).name)));
        fn = fieldnames(out.joints);
        for j = 1:numel(fn); out.joints.(fn{j}) = transpose(out.joints.(fn{j})); end
        fn = fieldnames(out.neutral);
        for j = 1:numel(fn); out.neutral.(fn{j}) = transpose(out.neutral.(fn{j})); end

        hz = out.hz_r; nfr = size(out.running.pelvis_1, 1);
        [ang, vel] = gait_kinematics(out.joints, out.neutral, out.running, hz, 0);

        RIC_FORCED_LABEL = 'run';
        [na, nv, ~, ~, ~, spd, evflag, ~] = gait_steps( ...
            out.neutral, out.running, ang, vel, hz, 0);
        RIC_FORCED_LABEL = '';

        nsL = size(na.L_ankle, 2); nsR = size(na.R_ankle, 2);
        evm = mean(evflag(:));

        curves = struct();
        for k = 1:numel(ANG); curves.(['ang_' ANG{k}]) = single(na.(ANG{k})); end
        for k = 1:numel(VEL); curves.(['vel_' VEL{k}]) = single(nv.(VEL{k})); end
        save(fullfile(outDir, [stem '.mat']), '-struct', 'curves', '-v7');
    catch ME
        RIC_FORCED_LABEL = '';
        errMsg = strrep(sprintf('%s: %s', ME.identifier, ME.message), ',', ';');
    end

    fprintf(mf, '%s,%s,%.1f,%.6f,%d,%d,%.4f,%.1f,%d,%.2f,%s\n', ...
            stem, subj, nominal, spd, nsL, nsR, evm, hz, nfr, toc(t0), errMsg);
    if mod(i, 5) == 0; fclose(mf); mf = fopen(manifestFile, 'a'); end

    if mod(i, 10) == 0 || i == n
        el = toc(t_start);
        fprintf('%d/%d  %.1f s/trial  elapsed %.1f min  eta %.1f min\n', ...
                i, n, el/i, el/60, (n-i)*el/i/60);
    end
end

fclose(mf);
fprintf('done: %d trials -> %s\n', n, manifestFile);
end
