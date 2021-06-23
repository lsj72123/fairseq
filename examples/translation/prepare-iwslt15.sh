#!/usr/bin/env bash

if [ ! -d "mosesdecoder" ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

if [ ! -d "subword-nmt" ]; then
  echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
  git clone https://github.com/rsennrich/subword-nmt.git
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=4000

SITE_PREFIX="https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi"

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=vi
lang=en-vi
prep=iwslt15.tokenized.en-vi
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig

echo "Download training dataset train.en and train.vi."
curl -o "train.en" "$SITE_PREFIX/train.en"
curl -o "train.vi" "$SITE_PREFIX/train.vi"

echo "Download dev dataset tst2012.en and tst2012.vi."
curl -o "valid.en" "$SITE_PREFIX/tst2012.en"
curl -o "valid.vi" "$SITE_PREFIX/tst2012.vi"

echo "Download test dataset tst2013.en and tst2013.vi."
curl -o "test.en" "$SITE_PREFIX/tst2013.en"
curl -o "test.vi" "$SITE_PREFIX/tst2013.vi"

echo "Download vocab file vocab.en and vocab.vi."
curl -o "vocab.en" "$SITE_PREFIX/vocab.en"
curl -o "$vocab.vi" "$SITE_PREFIX/vocab.vi"

cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.$l
    tok=train.tok.$l

    cat $orig/$f | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

perl $CLEAN -ratio 1.5 $tmp/train.tok $src $tgt $tmp/train.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.clean.$l > $tmp/train.$l
done

echo "pre-processing valid, test data..."
for l in $src $tgt; do
    for f in valid.$l test.$l; do
      cat $orig/$f | \
      perl $TOKENIZER -threads 8 -l $l | \
      perl $LC > $tmp/$f
      echo ""
    done
done


TRAIN=$tmp/train.$lang
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    echo "apply_bpe.py to train.$L..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE --dropout 0.1 < $tmp/train.$L > $tmp/bpe.train.$L
done

for L in $src $tgt; do
    for f in valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done