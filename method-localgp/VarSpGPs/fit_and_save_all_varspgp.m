% Run VarSpGPs

parpool(26)


% 1000 iterations

gp_mdl = varspgp_fit(x_train, y_train, 'n_xu', 100, 'n_iter', 1000, 'parallel', true);
save('output_emulator_varspgp_n_xu_100_n_iter_1000', 'gp_mdl')

gp_mdl = varspgp_fit(x_train, y_train, 'n_xu', 500, 'n_iter', 1000, 'parallel', true);
save('output_emulator_varspgp_n_xu_500_n_iter_1000', 'gp_mdl')

gp_mdl = varspgp_fit(x_train, y_train, 'n_xu', 1000, 'n_iter', 1000, 'parallel', true);
save('output_emulator_varspgp_n_xu_1000_n_iter_1000', 'gp_mdl')


% 5000 iterations

gp_mdl = varspgp_fit(x_train, y_train, 'n_xu', 100, 'n_iter', 5000, 'parallel', true);
save('output_emulator_varspgp_n_xu_100_n_iter_5000', 'gp_mdl')

gp_mdl = varspgp_fit(x_train, y_train, 'n_xu', 500, 'n_iter', 5000, 'parallel', true);
save('output_emulator_varspgp_n_xu_500_n_iter_5000', 'gp_mdl')

gp_mdl = varspgp_fit(x_train, y_train, 'n_xu', 1000, 'n_iter', 5000, 'parallel', true);
save('output_emulator_varspgp_n_xu_1000_n_iter_5000', 'gp_mdl')
