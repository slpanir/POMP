#!/bin/bash

binarize_fairseq_dataset(){
#    tgt=en
    raw=$WORKLOC/raw/train-set4test-langs
    bpe=$WORKLOC/bpe/train-set4test-langs
    fseq=$WORKLOC/fseq/train-set4test-langs
    mkdir -p $raw $bpe $fseq 

    for src in 'ne' 'si'; do
        for tgt in 'en'; do
          if [ $src != $tgt ]; then
#            mv $raw/${src}2${tgt}/$tgt $raw/${src}2${tgt}/train.$src-$tgt.$tgt
#            mv $raw/${src}2${tgt}/src $raw/${src}2${tgt}/train.$src-$tgt.$src
#            mkdir -p $bpe/${src}2${tgt}
            for f in "test.$src-$tgt.$src" "test.$src-$tgt.$tgt"; do
                python $WORKLOC/scripts/tools/spm_encode.py --model $WORKLOC/models/xlmrL_base/sentencepiece.bpe.model  \
                    --inputs $raw/$f --outputs $bpe/$f
            done
            python $WORKLOC/fairseq_cli/preprocess.py -s $src -t $tgt --dataset-impl lazy \
                --workers 24 --destdir $fseq --trainpref $bpe/train.$src-$tgt --testpref $bpe/test.$src-$tgt  \
                --srcdict  $WORKLOC/models/xlmrL_base/dict.txt \
                --tgtdict $WORKLOC/models/xlmrL_base/dict.txt
          fi
        done
    done 
}

sample_train_data(){
  raw=$WORKLOC/raw
  raw-zh=$WORKLOC/raw/zh-2000
  data=$WORKLOC/bpe
  bpe=$WORKLOC/bpe/train-2000
  fseq=$WORKLOC/fseq/train-2000
  # 确保目标目录存在
  mkdir -p $bpe $fseq ${raw-zh}

#  for tgt in de es fi hi ru zh; do
  for tgt in zh; do
#    cd $raw
#    seq 1 $(wc -l < "train.${tgt}-en.${tgt}") | shuf -n 2000 | perl -ne 'chomp; $lines{$_} = 1; END { open(F1, "<", "train.'$tgt'-en.'$tgt'") or die "Cannot open file train.'$tgt'-en.'$tgt': $!"; open(F2, "<", "train.'$tgt'-en.en") or die "Cannot open file train.'$tgt'-en.en: $!"; $count = 0; while (<F1>) { $count++; print if $lines{$count}; } continue { $_ = <F2>; print STDERR if $lines{$count}; } }' 1>zh-2000/train.${tgt}-en.${tgt} 2>zh-2000/train.${tgt}-en.en
#    for f in "train.${tgt}-en.${tgt}" "train.${tgt}-en.en"; do
#      python $WORKLOC/scripts/tools/spm_encode.py --model $WORKLOC/models/xlmrL_base/sentencepiece.bpe.model  \
#          --inputs ${raw-zh}/$f --outputs $bpe/$f
#    done
#    cd $data
    cd $WORKLOC
    python fairseq_cli/preprocess.py -s $tgt -t en --dataset-impl lazy \
            --workers 24 --destdir $fseq --trainpref $bpe/train.$tgt-en   \
            --srcdict  $WORKLOC/models/xlmrL_base/dict.txt \
            --tgtdict $WORKLOC/models/xlmrL_base/dict.txt
  done
}



export CUDA_VISIBLE_DEVICES=0
export WORKLOC=/mnt/e/unmt/acl22-sixtp

## First download the parallel corpora from urls in the appendix, put them in the $WORKLOC/raw path. All texts are supposed to be detokenized before running this script. 
## The dataset raw files are named with {train,valid,test}.{$src-en}.{$src,en}, e.g, train.de-en.de
## Assume the official XLM-R large model are stored in $WORKLOC/models/xlmrL_base 


binarize_fairseq_dataset
#sample_train_data