python fairseq_cli/train.py D:/datasets/iwslt15/en-vi/iwslt15_en_vi_cased/bin_th5 ^
--arch tfot_iwslt2015_en_vi ^
--share-decoder-input-output-embed ^
--optimizer adam ^
--adam-betas "(0.9, 0.98)" ^
--clip-norm 5.0 ^
--lr 5e-4 ^
--lr-scheduler inverse_sqrt ^
--warmup-updates 4000 ^
--dropout 0.2 ^
--weight-decay 0.0001 ^
--label-smoothing 0.1 ^
--max-tokens 5000 ^
--eval-bleu ^
--eval-bleu-args "{\"beam\":5, \"max_len_a\":1.2, \"max_len_b\":10}" ^
--eval-bleu-remove-bpe ^
--eval-bleu-detok moses ^
--eval-bleu-print-samples ^
--best-checkpoint-metric bleu ^
--maximize-best-checkpoint-metric ^
--max-update 20000 ^
--log-file log.txt ^
--source-lang en ^
--target-lang vi ^
--log-format simple ^
--sentence-avg ^
--save-dir ./checkpoint/lstm-ot-en-vi-ot ^
--criterion label_smoothed_cross_entropy_with_ot ^
--ot-weight-match 1.0