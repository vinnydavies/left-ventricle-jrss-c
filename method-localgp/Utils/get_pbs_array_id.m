function id = get_pbs_array_id()

id = str2double( getenv('PBS_ARRAYID') );

end