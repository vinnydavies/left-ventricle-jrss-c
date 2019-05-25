function title_string = make_title(name)

[name, data_string] = split_name(name, 'Data');
[name, method_string] = split_name(name, 'Method');
[name, distance_string] = split_name(name, 'Distance');
[name, objective_string] = split_name(name, 'Objective');
[name, cov_string] = split_name(name, 'Cov');
[name, emulation_string] = split_name(name, 'Emulation');

title_string = strjoin({emulation_string cov_string objective_string method_string}, ', ');

end


function [name, final_string] = split_name(name, word)

% Get indices
[start_id, end_id] = regexp(name, [word '\w*']);
% Find substring without space
final_string = name(start_id:end_id);
% Split words with space
final_string = insertAfter(final_string, word, ' ');
% Delete couple of words
name = regexprep(name, ['_' word '\w*'], '');

end