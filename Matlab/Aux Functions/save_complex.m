function save_complex(path, A)

    % Writes the complex matrix number to two files _r and _i, in binary
    
    % Write real part
    fid = fopen([path, '_r.bin'],'wb'); 
    fwrite(fid, real(A), 'single');
    fclose(fid);
    
    % Write imaginary part
    fid = fopen([path, '_i.bin'],'wb'); 
    fwrite(fid, imag(A), 'single');
    fclose(fid);


end

