function [instance_info_out] = format_instance_info(instance_info_in)
    
    % The first position is used for the parallelisation level [0-3]
    % this parameter is used along with parameters 2 to determine the
    % amount of builders to create and which builder to load and use in the
    % channel calculation phase
    % The second parameter has 2 functions, depending    
    
    % 1- in the setup/save_setup fase to tell matlab in how many chunks
    % it needs to divide each user's track in order to improve parallel
    % performance, i.e. number of divisions in time. 
    % The number of instances is derived from this and the 1st parameter.
    % This will be the number of builder sets.
    
    % 2- in the channel calculation phase, to tell matlab the high level
    % instance to use for the computation. Here, the first parameters comes
    % into play to derive which file to load and which builder to use.
    
    if ischar(instance_info_in) || isstring(instance_info_in)
        % String treatment for triming variable
        s_aux = split(instance_info_in, ',');

        if length(s_aux) == 1
            error('Instance_info should be separated by comma.');
        end

        string_1 = char(s_aux(1));
        string_1 = string_1(2:end);
        string_2 = char(s_aux(2));
        string_2 = string_2(1:end-1);

        if isnan(str2double(string_1)) || isnan(str2double(string_2))
            error('The format should be, e.g, [0,0.1]');
        end

        instance_info_out = [str2double(string_1), str2double(string_2)];
    else
        if size(instance_info_in, 2) ~= 2
            error(['Only expecting 2 inputs for simulation trim, not ', ...
            num2str(size(instance_info_in, 2)), '.']);
        else
            instance_info_out = instance_info_in;
        end
    end
    
end