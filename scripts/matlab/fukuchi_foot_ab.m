function fukuchi_foot_ab(codeFolder, jsonDir, outDir, manifestFile, deriveFoot)
%FUKUCHI_FOOT_AB Run the Fukuchi batch with the foot long axis assumed or measured.
%
% deriveFoot = 0 -> the pipeline's original hardcoded axis (zero toe-out).
% deriveFoot = 1 -> measured per subject from heel centroid to metatarsal midpoint.
%
% With deriveFoot = 0 the output MUST be bit-identical to the pre-patch batch.
% That is the regression test: the opt-in patch must not have touched the default
% path, because every existing result in this repository came through it.

addpath(codeFolder);
global RIC_FORCED_LABEL RIC_DERIVE_FOOT_AXIS %#ok<GVMIS>

if ~exist(outDir, 'dir'); mkdir(outDir); end

mf = fopen(manifestFile, 'w');
fprintf(mf, ['trial,subject,speed_nominal,speed_computed,nsteps_L,nsteps_R,' ...
             'eventsflag,hz,n_frames,secs,err\n']);

files = dir(fullfile(jsonDir, 'RBDS*run*.json'));
files = files(~contains({files.name}, '.meta.'));
n = numel(files);
fprintf('%d trials | deriveFoot = %d\n', n, deriveFoot);

ANG = {'L_ankle','L_knee','L_hip','L_foot','L_pelvis', ...
       'R_ankle','R_knee','R_hip','R_foot','R_pelvis'};
VEL = {'L_ankle','L_knee','L_hip','L_pelvis','R_ankle','R_knee','R_hip','R_pelvis'};

t_start = tic;
for i = 1:n
    [~, stem] = fileparts(files(i).name);
    tok = regexp(stem, '^(RBDS\d+)runT(\d+)$', 'tokens');
    if isempty(tok); continue; end
    subj = tok{1}{1};
    nominal = str2double(tok{1}{2}) / 10;

    t0 = tic; errMsg = ''; spd = NaN; nsL = NaN; nsR = NaN; evm = NaN;
    hz = NaN; nfr = NaN;
    try
        out = jsondecode(fileread(fullfile(jsonDir, files(i).name)));
        fn = fieldnames(out.joints);
        for j = 1:numel(fn); out.joints.(fn{j}) = transpose(out.joints.(fn{j})); end
        fn = fieldnames(out.neutral);
        for j = 1:numel(fn); out.neutral.(fn{j}) = transpose(out.neutral.(fn{j})); end

        hz = out.hz_r; nfr = size(out.running.pelvis_1, 1);

        RIC_DERIVE_FOOT_AXIS = deriveFoot;
        [ang, vel] = gait_kinematics(out.joints, out.neutral, out.running, hz, 0);
        RIC_FORCED_LABEL = 'run';
        [na, nv, ~, ~, ~, spd, evflag, ~] = gait_steps( ...
            out.neutral, out.running, ang, vel, hz, 0);
        RIC_FORCED_LABEL = ''; RIC_DERIVE_FOOT_AXIS = [];

        nsL = size(na.L_ankle, 2); nsR = size(na.R_ankle, 2);
        evm = mean(evflag(:));
        curves = struct();
        for k = 1:numel(ANG); curves.(['ang_' ANG{k}]) = single(na.(ANG{k})); end
        for k = 1:numel(VEL); curves.(['vel_' VEL{k}]) = single(nv.(VEL{k})); end
        save(fullfile(outDir, [stem '.mat']), '-struct', 'curves', '-v7');
    catch ME
        RIC_FORCED_LABEL = ''; RIC_DERIVE_FOOT_AXIS = [];
        errMsg = strrep(sprintf('%s: %s', ME.identifier, ME.message), ',', ';');
    end

    fprintf(mf, '%s,%s,%.1f,%.6f,%d,%d,%.4f,%.1f,%d,%.2f,%s\n', ...
            stem, subj, nominal, spd, nsL, nsR, evm, hz, nfr, toc(t0), errMsg);
    if mod(i, 10) == 0
        fclose(mf); mf = fopen(manifestFile, 'a');
        fprintf('%d/%d  elapsed %.1f min\n', i, n, toc(t_start)/60);
    end
end

fclose(mf);
fprintf('done: %d trials (deriveFoot=%d) -> %s\n', n, deriveFoot, manifestFile);
end
