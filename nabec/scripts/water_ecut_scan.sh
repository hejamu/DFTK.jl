#!/usr/bin/env bash
# H2O at L=12, Γ: NABEC (Sternheimer) vs FD dipole APT as a function of Ecut.
cd ~/WORK/playground/naBEC
for E in 20 30 40; do
  NABEC_L=12 NABEC_NK=1 NABEC_ECUT=$E NABEC_METHOD=stn srun -p Workstations --exclude=ws05 -c 16 --mem=32G -t 04:00:00 \
    ~/.juliaup/bin/julia --project=. -t 16 test/test_water.jl 2>&1 | grep -E "^L=|n_G|time|rror" | sed "s/^/Ecut=$E /"
done
