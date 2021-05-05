function A = read_complex(path, size_output)

    % Reads the complex matrix from two files _r and _i
    % size_output is the dimensions of the original matrix
    
    % Read real part
    fid = fopen([path, '_r.bin'],'r'); 
    c_read_r = reshape(fread(fid, 'single=>single'), size_output);
    fclose(fid);
           
    % Read imaginary part
    fid = fopen([path, '_i.bin'],'r'); 
    c_read_i = reshape(fread(fid, 'single=>single'), size_output);
    fclose(fid);
    
    % Put them together
    A = c_read_r + 1i* c_read_i;
end

