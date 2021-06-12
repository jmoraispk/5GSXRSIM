function [] = disp_pattern_cut_information(cut_pattern, ang_vals)
    
    % Given a cut (elevation or azimuth), finds:
    % Max directivity, HPBW, FNBW, SLL
    % Inputs: directivity values along that cut, angular values
    % when small_idx and large_idx are mentioned, we are talking about
    % lobes or nules around the main lobe, and the small/large refers to the
    % previous/posterior index with a null/side lobe. Hence, small_idx is
    % smaller than the index of the main lobe, and vice-versa for large.
                             
    % Finds directivity in elevation cut and idx of main lobe.
    [max_dir, max_idx] = max(cut_pattern);
    
    % Find HPBW:
    hpbw_small_idx = find(cut_pattern > max_dir - 3, 1, 'first');
    hpbw_large_idx = find(cut_pattern > max_dir - 3, 1, 'last');
    HPBW = ang_vals(hpbw_large_idx) - ang_vals(hpbw_small_idx);
    
    
    % Find FNBW:
    null_large_idx = find(islocalmin(cut_pattern(max_idx+1:end)), 1, 'first');
	null_small_idx = find(islocalmin(cut_pattern(1:max_idx-1)), 1, 'last');
    null_large_idx = null_large_idx + numel(1:max_idx);
    FNBW = ang_vals(null_large_idx) - ang_vals(null_small_idx);
    if FNBW < HPBW
        FNBW = 0;
    end
    % Find SLL
    sl_large_idx = find(islocalmax(cut_pattern(max_idx:end)), 1, 'first');
    sl_small_idx = find(islocalmax(cut_pattern(1:max_idx)), 1, 'last');
    sl_large_idx = sl_large_idx + numel(1:max_idx-1);
        % Check which is bigger and report that SLL
    if cut_pattern(sl_large_idx) > cut_pattern(sl_small_idx)
        SLL = cut_pattern(sl_large_idx) - max_dir;
    else
        SLL = cut_pattern(sl_small_idx) - max_dir;
    end
    
    
    % disp: directivity, HPBW, FNWB, SLL
    fprintf('Directivity: %2.1f dBi\n', max_dir);
    fprintf('SLL: %12.1f dB\n', SLL);
    fprintf('HPBW: %4.1fº\n', HPBW);
    fprintf('FNBW: %4.1fº\n', FNBW);
end

