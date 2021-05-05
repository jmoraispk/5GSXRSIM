function [] = compilerv3(testing)
    % Compiles n instances with log names from 1 to n
    % and places them in the 'Instances' directory, one in each folder
    instance_home_directory = 'Instances-Home';
    
    working_dir = pwd();
    
    if ~strcmp(working_dir(end-5:end), 'Matlab')
        disp('WARNING: This compiler only works in Matlab''s root folder');
        return
    end
    
    disp('Compile to run in remote serve notice:');
    disp(['If this is supposed to run in a machine that does not ', ...
          'have Matlab installed, then first installing Matlab ', ...
          'Runtime is required. To do this, use this link and select ', ...
          'the Matlab version you''re using, and the machine''s ', ...
          'operative system:', newline, ...
          'https://nl.mathworks.com/products/compiler/matlab-runtime.html']);
    
    if testing
        filename = 'test4';
        
        mcc('-m', [filename, '.m'], '-R', '-logfile', '-R', 'log.txt');

    else
        filename = 'Meeting12';
        
        mcc('-m', [filename, '.m'], ...
            '-a', './QuaDRiGa/*', '-a', './Aux Functions/*', ...
            '-R', '-logfile', '-R', 'log.txt');

    end
    
    disp('Done Compiling.'); 
    filename = [filename, '.exe'];
    
    mkdir(instance_home_directory);
    movefile(filename, instance_home_directory);
    
    % Clean main folder
    delete('readme.txt');
    delete('requiredMCRProducts.txt');
    delete('mccExcludedFiles.log');
    
    disp('Done Organising Instances.');
end