python -m rouge.rouge \
        --target_filepattern=../feb14/beam_max1865_feb14_notuselatent_0_gt_4.txt \
            --prediction_filepattern=../feb14/beam_max1865_feb14_notuselatent_0_gen_4.txt \
                --output_filename=scores.csv \
                    --use_stemmer=true \
                        --split_summaries=true
