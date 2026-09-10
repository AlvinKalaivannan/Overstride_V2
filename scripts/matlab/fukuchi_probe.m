function fukuchi_probe(codeFolder, jsonFile, outFile)
%FUKUCHI_PROBE Run the RIC pipeline on one adapted Fukuchi trial.
%
% Stage-1 sanity check for phase 10. Fukuchi has no stored dv_r, so the
% two-label reconciliation the archive batches use is unavailable -- the label
% is forced to 'run', which is what these trials are.
%
% Prints the sagittal range for each joint. Running should give roughly:
%   hip flexion  ~40-50 deg range,  knee ~70-85 deg,  ankle ~45-55 deg
% Anything far outside that means the adapter is wrong, and it is better to
% find that here than after processing 100 trials.

addpath(codeFolder);
global RIC_FORCED_LABEL %#ok<GVMIS>

raw = fileread(jsonFile);
out = jsondecode(raw);

% JSON does not faithfully recreate joints/neutral -- transpose, exactly as
% processing_code_example.m:55 and the archive batches do.
fn = fieldnames(out.joints);
for j = 1:numel(fn); out.joints.(fn{j}) = transpose(out.joints.(fn{j})); end
fn = fieldnames(out.neutral);
for j = 1:numel(fn); out.neutral.(fn{j}) = transpose(out.neutral.(fn{j})); end

fprintf('frames: %d @ %g Hz\n', size(out.running.pelvis_1, 1), out.hz_r);

[ang, vel] = gait_kinematics(out.joints, out.neutral, out.running, out.hz_r, 0);

RIC_FORCED_LABEL = 'run';
[na, nv, ~, ~, DV, spd, evflag, ~] = gait_steps( ...
    out.neutral, out.running, ang, vel, out.hz_r, 0);
RIC_FORCED_LABEL = '';

fprintf('speed: %.3f m/s | steps L %d R %d | eventsflag %.3f\n', ...
        spd, size(na.L_ankle, 2), size(na.R_ankle, 2), mean(evflag(:)));

% Plane 3 (index 3) is flexion/extension -- gait_kinematics documents the
% segment frame as "Z points to the subject's right side [Hinge flexion
% extension]", and CLAUDE.md records the same finding measured against dv_r.
joints = {'R_hip', 'R_knee', 'R_ankle', 'L_hip', 'L_knee', 'L_ankle'};
fprintf('\n%-8s %10s %10s %10s\n', 'joint', 'min', 'max', 'range');
for k = 1:numel(joints)
    c = mean(na.(joints{k})(:, :, 3), 2);      % mean across steps, sagittal
    fprintf('%-8s %10.1f %10.1f %10.1f\n', joints{k}, min(c), max(c), max(c) - min(c));
end

curves = struct();
names = {'L_ankle','L_knee','L_hip','L_foot','L_pelvis', ...
         'R_ankle','R_knee','R_hip','R_foot','R_pelvis'};
for k = 1:numel(names)
    curves.(['ang_' names{k}]) = single(na.(names{k}));
end
vnames = {'L_ankle','L_knee','L_hip','L_pelvis','R_ankle','R_knee','R_hip','R_pelvis'};
for k = 1:numel(vnames)
    curves.(['vel_' vnames{k}]) = single(nv.(vnames{k}));
end
curves.speed = spd;
curves.eventsflag = mean(evflag(:));
save(outFile, '-struct', 'curves', '-v7');
fprintf('\nwrote %s\n', outFile);
end
