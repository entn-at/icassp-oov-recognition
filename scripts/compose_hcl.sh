#!/bin/bash -e

work=$1
lex=$2
tree=$3
model=$4
isym=$5
osym=$6

N=$(tree-info $tree | grep "context-width" | cut -d' ' -f2) || { echo "Error when getting context-width"; exit 1; }
P=$(tree-info $tree | grep "central-position" | cut -d' ' -f2) || { echo "Error when getting central-position"; exit 1; }

rm -rf $work
mkdir $work

L=$work/L.fst
#utils/lang/make_lexicon_fst.py --sil-prob=0.5 --sil-phone=SIL $lex > ${L}.txt
python create_lfst.py $lex $isym $osym $L 

#fstcompile --isymbols=$isym --osymbols=$osym ${L}.txt | fstaddselfloops disambig_in disambig_out | fstarcsort --sort_type=ilabel > $L
#fstcompile --isymbols=$isym --osymbols=$osym ${L}.txt | fstarcsort --sort_type=ilabel > $L

fstdeterminizestar $L | fstarcsort --sort_type=ilabel > $work/L_det.fst

fstmakecontextfst --read-disambig-syms=data_cv_word/lang_static/phones/disambig.int --central-position=$P --context-size=$N $isym 303 $work/ilabels_${N}_${P} > $work/C.fst

#fstcomposecontext --context-size=$N --central-position=$P \
#    --read-disambig-syms=disambig_in --write-disambig-syms=$work/disambig_new \
#    $work/ilabels_${N}_$P $work/L_det.fst | fstarcsort --sort_type=ilabel > $work/CL_tmp.fst
fstcompose $work/C.fst $work/L_det.fst | fstarcsort > $work/CL_tmp.fst

make-h-transducer --disambig-syms-out=$work/disambig_tid.int --transition-scale=1.0 $work/ilabels_${N}_${P} \
    $tree $model | fstarcsort --sort_type=olabel > $work/Ha.fst

fstcompose $work/Ha.fst $work/CL_tmp.fst | fstdeterminizestar | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=olabel > $work/HCL.fst
