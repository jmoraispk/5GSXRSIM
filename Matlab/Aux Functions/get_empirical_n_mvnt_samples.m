function [n_samples] = get_empirical_n_mvnt_samples(simulation_duration, ...
                                                    r_head_sphere)
    
    % Just a random number of samples for creating the positions of the
    % track. Most of them should get thrown away right after the
    % interpolation. It only serves the purpose of having enough of them
    % to create the track.
                                                
    %adjustment constant for movement samples generation
    if simulation_duration < 1
        adjust_const = 5;
    else
        adjust_const = 1;
    end

    %movement samples. This formula is needed to generate samples in excess
    n_samples = 20 * simulation_duration / (2 * r_head_sphere) ...
                     * adjust_const;
    if n_samples < 2
        n_samples = 2;
    end
    
end
    