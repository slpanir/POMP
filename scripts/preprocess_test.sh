#!/bin/bash

binarize_fairseq_dataset(){
    tgt=en
    raw=$WORKLOC/raw 
    bpe=$WORKLOC/bpe 
    fseq=$WORKLOC/fseq
    #mkdir -p $raw $bpe $fseq 

    for src in 'lv' 'ne' 'si' 'gu' 'et'; do 
#        for f in "test.$src-en.$src" "test.$src-en.en"; do 
#            python scripts/tools/spm_encode.py --model $WORKLOC/models/xlmrL_base/sentencepiece.bpe.model  \
#                --inputs $raw/$f --outputs $bpe/$f
#        done 
        python fairseq_cli/preprocess.py -s $src -t $tgt --dataset-impl lazy \
            --workers 24 --destdir $fseq --testpref $bpe/test.$src-en   \
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
