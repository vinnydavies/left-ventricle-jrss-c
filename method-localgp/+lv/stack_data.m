function data = stack_data(name, n)

for i = 1:n
    tmp = load( sprintf('%s%d', name, i) );
    volume = tmp.LVVolumeMRI(:)';
    strain = tmp.strainMRITotal(:)';
    data(i,:) = [volume, strain];
end

end