#!/bin/zsh

lm-pipeline -d data/populist/ds.pkl -m text-davinci-002
# lm-pipeline -d data/badelite1/ds.pkl -m text-davinci-002
# lm-pipeline -d data/badelite2/ds.pkl -m text-davinci-002
# lm-pipeline -d data/goodpeople/ds.pkl -m text-davinci-002
# lm-pipeline -d data/badeliterationalemini/ds.pkl -m text-davinci-002 --n_tokens 100 --n_probs 100 --temperature 0.0 --stop '\n\n' --ix_to_check -1 -n 2
# lm-pipeline -d data/goodpeoplerationale/ds.pkl -m text-davinci-002 --n_tokens 100 --n_probs 100 --temperature 0.0 --stop '\n\n'
# lm-pipeline -d data/goodpeoplerationale/ds.pkl -m text-davinci-002