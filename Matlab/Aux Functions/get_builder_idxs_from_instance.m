function [time_division_idx, builder_idxs] = get_builder_idxs_from_instance(...
                instance_info_param, n_instances_per_time_division, ...
                n_builders_per_time_div)
            
            
    % Gets the time division index (the set of builders)
    % and the builder indices inside that builder set.
    % Note, these builder indices go from 1 to n_builders_per_time_division
     
    
    time_division_idx = floor((instance_info_param-1) / ...
                            n_instances_per_time_division)+1;
                        
                        
    % Given the set of n_builders, how to select the ones we want?
        
    % The number of builders per instance is going to be the length of
    % builder_idxs. Note: there's a different set of builders for each
    % time division. So, we divide the number of buiders in a time
    % divisions by the number of instances in a time division.
    % DO WE NEED THIS:The round is only to make from double to integer,
    % even though the division will always be a whole number
    n_builders_per_instance = n_builders_per_time_div ...
                              / n_instances_per_time_division;

    % DEPENDING on the number of instances per time division
    % they may have more than one builder
    builder_idxs = 1:n_builders_per_instance;
    % And they may not start and index 1, but further ahead, that
    % compensation is made
    inst_idx_on_time_division = mod(instance_info_param, ...
                                    n_instances_per_time_division) - 1;
    %the problem with not zero-indexing
    if inst_idx_on_time_division == -1
        inst_idx_on_time_division = n_instances_per_time_division - 1;
    end
    builder_idxs = builder_idxs + n_builders_per_instance * ...
                                  inst_idx_on_time_division;

end

