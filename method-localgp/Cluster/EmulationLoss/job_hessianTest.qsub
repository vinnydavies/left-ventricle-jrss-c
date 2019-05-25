#!/bin/bash
#PBS -N hessianTest
#PBS -q defaultq
#PBS -l nodes=1:ppn=24
#PBS -j oe
#PBS -l walltime=10:00:00
#PBS -t 1-100

/maths/MATLAB/R2017a/bin/matlab -nodisplay -nosplash -r "\
cd('~/Documents/MATLAB/LeftVentricleProject'); \
startup; \
lv.run_hessian_loss( 'log_loss', 'test', get_pbs_array_id() ); \
exit()"