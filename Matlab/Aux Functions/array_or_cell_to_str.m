function [s] = array_or_cell_to_str(arr, sep)

    % Returns a string of the whole array, maximum 2 dimensions
    
    
    if isempty(sep)
        sep = ' ';
    end
    s = '';
    if iscell(arr)
        for i = 1:size(arr, 1)
            for j = 1:size(arr, 2)
                if size(arr{i, j}, 1) > 1 
                    s_aux = array_or_cell_to_str(arr{i,j});
                else
                    s_aux = num2str(arr{i,j});
                end
                
                if i == 1 && j == 1
                    s = s_aux;
                else
                    s = [s, sep, s_aux];
                end
            end
        end
    else
        for row = 1:size(arr, 1)
            if row == 1
                s = num2str(arr(row, :)); %#ok<*AGROW>
            else
                s = [s, sep, num2str(arr(row, :))];
            end
        end
    end

end

