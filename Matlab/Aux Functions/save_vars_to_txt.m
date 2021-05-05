function [] = save_vars_to_txt(txt_path, vars_names, vars_values)

    % Saves variables to text file, with their names
    % The main purpose of this function is to save some key variables to
    % a text file, such that you may easily identify a long simulation
    % simply by opening a .txt

    % NOTE: vars_names and vars_values must be cell arrays!

    if length(vars_names) ~= length(vars_values)
        error('Names much match values');
    end

    fid = fopen(txt_path, 'wt');

    for var_idx = 1:length(vars_names)
        
        % disp(vars_names{var_idx});
        if size(vars_values{var_idx}, 1) > 1 || iscell(vars_values{var_idx})
            s = [vars_names{var_idx}, ' = ', ...
                      array_or_cell_to_str(vars_values{var_idx}, '; '), ...
                      newline];
        else
            s = [vars_names{var_idx}, ' = ', ...
                      num2str(vars_values{var_idx}), newline];
        end
        % disp(s);
        fprintf(fid, s);
    end
    
    fclose(fid);
end

