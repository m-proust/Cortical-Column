#!/bin/zsh
# Cosyne abstract Fig. 2D: finite-size scaling of the L6 alpha, with independent or partly shared drive.
# Isolated L6 scaled to N x k cells (conditions 'L6_Nx<k>', same in-degrees, Poisson N of L6).
#   independent/  every E cell has its own 45 Poisson sources            (input correlation c = 0)
#   shared/       n of each E cell's 45 sources are common to all E cells (c = n/45: 2, 4, 11, 20 %)
# Mean and variance of each cell's drive are unchanged; rates are NOT re-matched.
# 10 seeds per point, 4 runs in parallel. Resumable.
#   zsh studies/cosyne_abstract/run_finite_size.sh
cd "$(dirname "$0")/../.."
OUT=saved_trials/cosyne/finite_size
mkdir -p $OUT/independent $OUT/shared $OUT/logs
for k in 8 4 2 1 0.5; do for n in 0 1 2 5 9; do for s in 1 2 3 4 5 6 7 8 9 58925; do
  if [ $n = 0 ]; then tag=independent/L6_Nx${k}_s${s}; else tag=shared/L6_Nx${k}_sh${n}_s${s}; fi
  [ -f $OUT/$tag.npz ] || echo "$k $n $s $tag"
done; done; done | xargs -P 4 -L 1 zsh -c \
  '~/Desktop/venv/bin/python minimal_config/run_spectrum.py L6_Nx$0 '$OUT'/$3.npz --seed $2 --shared-E $1 > '$OUT'/logs/${3:t}.log 2>&1'
echo done
