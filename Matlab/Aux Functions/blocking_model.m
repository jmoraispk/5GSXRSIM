function [c_blocked] = blocking_model(c, blocking_mode)

    % Receives a channel between a transmitter and a receiver, at a 
    % given frequency, with coefficients and path information of a 
    % certain interval in time, and modifies the channel to simulate 
    % human blocking in a meeting room.

    %%%%%%%%%%%%% Description of each blocking model %%%%%%%%%%%%%%%%%%%
    
    % 1 - Testing only: Halves the amplitudes of each coefficient
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % This is for testing only: Puts all coeffs' amplitude to half!
    if blocking_mode == 1
        % if we decide to have a copy.
        c_blocked = c.copy();
        
        % Dividing by 2 in the linear (electric field) scale, is -6dB in
        % the power scale.
        c_blocked.coeff(:) = c.coeff(:) ./ 2;
    end
    
    
    
    
    
    
    % NOTE: in case the copy is required, c_blocked is returned
    %       automatically. If the copy is not required, then include the
    %       line below. (Not doing a copy saves memory.)
    % c_blocked = c;
end

