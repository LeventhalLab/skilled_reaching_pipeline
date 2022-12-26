% calculate fundamental matrices from matched points in mouse skilled
% reaching videos

parent_directory = 'C:\Users\dklev\Dropbox (University of Michigan)\MED-LeventhalLab\Burgess_data\mouse_SR_videos_to_analyze\mouse_SR_videos_tocrop\';

mouse_dirs = dir(fullfile(parent_directory,'*'));

for i_mousedir = 1 : length(mouse_dirs)
    if strcmp(mouse_dirs(i_mousedir).name, '.') || strcmp(mouse_dirs(i_mousedir).name, '..')
        continue
    end
    mouse_path = fullfile(mouse_dirs(i_mousedir).folder, mouse_dirs(i_mousedir).name);
    if ~isfolder(mouse_path)
        continue
    end
    mouseID = mouse_dirs(i_mousedir).name;

    month_dirs = dir(fullfile(mouse_path, sprintf('%s_*', mouseID)));

    for i_mdir = 1 : length(month_dirs)
        mdir = fullfile(month_dirs(i_mdir).folder, month_dirs(i_mdir).name);
        if ~isfolder(mdir)
            continue
        end

        session_dirs = dir(fullfile(mdir, sprintf('%s*', month_dirs(i_mdir).name)));

        for i_sdir = 1 : length(session_dirs)

            sdir = fullfile(session_dirs(i_sdir).folder, session_dirs(i_sdir).name);
            if ~isfolder(sdir)
                continue
            end

            fund_mat_from_mouse_vid_folder(sdir);

        end

    end

end