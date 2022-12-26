function [F] = fund_mat_from_mouse_vid_folder(session_folder)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[~,session_date,~] = fileparts(session_folder);

num_sessions_this_date = 0;
vid_list = dir(fullfile(session_folder, sprintf('%s_*_cam*.avi', session_date)));

vid_session_nums = zeros(length(vid_list), 1);
for i_vid = 1 : length(vid_list)
    name_parts = split(vid_list(i_vid).name, '_');
    vid_session_nums(i_vid) = str2double(name_parts{4});
end
vid_session_nums = unique(vid_session_nums);

% find a cam01, cam02 pair for each session

for i_session = 1 : length(vid_session_nums)

    % take the first vid from each session; will have to see if those are
    % ever missing...
    vid_list = dir(fullfile(session_folder, sprintf('%s_*_%d_000_cam*.avi', session_date, vid_session_nums(i_session))));
    F = fund_mat_from_mouse_vids(vid_list);
end

end