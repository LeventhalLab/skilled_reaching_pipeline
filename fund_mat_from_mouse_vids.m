function [F] = fund_mat_from_mouse_vids(vid_list)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

num_cams = length(vid_list);  % should be 2
cam_nums = zeros(num_cams,1);
vid_nums = zeros(num_cams,1);
for i_cam = 1 : num_cams

    % get the camera number for each video
    name_parts = split(vid_list(i_cam).name, '_');
    cam_nums(i_cam) = str2num(name_parts{6}(4:5));
    vid_nums(i_cam) = str2num(name_parts{5});

end

if ~all(vid_nums == vid_nums(1))
    % not all video numbers are the same
    error('video numbers do not match')
end

for i_cam = 1 : num_cams
    
    vid_name = fullfile(vid_list(i_cam).folder, vid_list(i_cam).name);

    % read the first frame
    vidObj = VideoReader(vid_name);

    vf = readFrame(vidObj);
    vidFrame{i_cam} = rgb2gray(vf);

    if cam_nums(i_cam) == 1
        vidFrame{i_cam} = imrotate(vidFrame{i_cam}, 180);
    end

end

for match_threshold = 0.2:0.2:2
    points1 = detectSURFFeatures(vidFrame{1});
    points2 = detectSURFFeatures(vidFrame{2});
    
    [f1,vpts1] = extractFeatures(vidFrame{1},points1);
    [f2,vpts2] = extractFeatures(vidFrame{2},points2);
    
    indexPairs = matchFeatures(f1,f2, MatchThreshold=match_threshold);
    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));
    
    figure; showMatchedFeatures(vidFrame{1},vidFrame{2},matchedPoints1,matchedPoints2);
    legend("matched points 1","matched points 2");

    title(sprintf('match threshold = %d', match_threshold));

end

end