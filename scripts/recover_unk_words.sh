#!/bin/bash -e

lats=$1
work=$2
lang=$3
mdl=$4
lsym=$5  #letter
psym=$6  #phone
lmwt=$7
wip=$8
out=$9

lattice-align-words $lang/phones/word_boundary.int $mdl "ark:gunzip -c $lats/lat*gz |" ark:- | \
    lattice-scale --inv-acoustic-scale=$lmwt ark:- ark:- | \
    lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- | \
    lattice-1best ark:- ark:- | \
    lattice-arc-post --acoustic-scale=0.1 $mdl ark:- - | \
    utils/int2sym.pl -f 5 $lang/words.txt | \
    utils/int2sym.pl -f 6- $lang/phones.txt > $work/post_${lmwt}_${wip}.txt

echo "Gotten posts"

grep '<unk>' $work/post_${lmwt}_${wip}.txt | cut -d' ' -f 1,6- | sed -r 's/(_B|_I|_S|_E)//g'> $work/unk_phone_arcs_${lmwt}_${wip}.txt

python $CODE/condutor/scripts/create_lca.py -read-syms-f $psym -isark $work/unk_phone_arcs_${lmwt}_${wip}.txt $work/unk_phone_fsts_${lmwt}_${wip}.ark

fsts-compose ark:$work/unk_phone_fsts_${lmwt}_${wip}.ark libri_g2p/p2g_model.fst ark:$work/letter_fsts_${lmwt}_${wip}.ark

python expand_fsts.py  $work/letter_fsts_${lmwt}_${wip}.ark to_expand $lsym $work/letter_fsts_exp_${lmwt}_${wip}.ark
#python expand_fsts.py -noexpand $work/unk_phone_fsts_${lmwt}_${wip}.ark "" $lsym $work/letter_fsts_exp_${lmwt}_${wip}.ark

fsts-compose ark:$work/letter_fsts_exp_${lmwt}_${wip}.ark cv_char_lm/char_o8.fst ark:- | \
    fsts-to-transcripts ark:- ark,t:- | sed -r 's/(179|180|181)//g' | utils/int2sym.pl -f 2- $lsym |\
    awk '{printf $1" "; for(i=2;i<=NF;i++) {printf $i} printf "\n"}' > $out 
