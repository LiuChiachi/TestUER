CUDA_VISIBLE_DEVICES="5"  nohup  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/afqmc_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/afqmc_public/dev.json  \
                                        --learning_rate_list  1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 3   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 >  6_6l_afqmc.log &

CUDA_VISIBLE_DEVICES="5"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/tnews_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/tnews_public/dev.json  \
                                        --learning_rate_list  1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 3   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 >  6_6l_tnews.log &

CUDA_VISIBLE_DEVICES="6"  nohup python3  -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path  /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/iflytek_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/iflytek_public/dev.json  \
                                        --learning_rate_list 1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 6   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > 6_6l_ifly.log &

CUDA_VISIBLE_DEVICES="4"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/ocnli_public/train.50k.json --dev_path /root/.paddlenlp/datasets/Clue/ocnli_public/dev.json  \
                                        --learning_rate_list  1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 6   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 >  6_6l_ocnli.log &

CUDA_VISIBLE_DEVICES="5"  nohup  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/csl_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/csl_public/dev.json  \
                                        --learning_rate_list  1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 8   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > 6_6l_csl_log &

CUDA_VISIBLE_DEVICES="4"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/cmnli_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/cmnli_public/dev.json  \
                                        --learning_rate_list  1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 3   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible  --seq_length 128 >  6_6l_cmnli.log &

CUDA_VISIBLE_DEVICES="4"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/XNLI/xnli/train/part-0 --dev_path /root/.paddlenlp/datasets/XNLI/xnli/dev/part-0  \
                                        --learning_rate_list  1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 3   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 >  6_6l_xnli.log &

CUDA_VISIBLE_DEVICES="5"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/ChnSentiCorp/ChnSentiCorp/ChnSentiCorp/train.tsv --dev_path /root/.paddlenlp/datasets/ChnSentiCorp/ChnSentiCorp/ChnSentiCorp/dev.tsv  \
                                        --learning_rate_list  1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 6   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 >  6_6l_chnsenticorp.log &

CUDA_VISIBLE_DEVICES="6"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/6l768d/cluecorpussmall_roberta_L-6_H-768_seq512_model.bin-250000  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/6l768d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/6l768d/config.json \
                                        --train_path /root/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/train.tsv --dev_path /root/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/dev.tsv  \
                                        --learning_rate_list  1e-4 5e-5 3e-5 --batch_size_list 16 32 64 --epochs_num_list 3   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 >  6_6l_lcqmc.log &

