CUDA_VISIBLE_DEVICES="0" python3 finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/afqmc_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/afqmc_public/dev.json  \
                                        --learning_rate_list 1e-4 5e-5 3e-5 --batch_size_list 32 64 --epochs_num_list 3   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible  --seq_length 128 > search_2_tx_afqmc.log &



CUDA_VISIBLE_DEVICES="0" python3 finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/tnews_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/tnews_public/dev.json  \
                                        --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list  32 64 --epochs_num_list 3 5 8  \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128  > search_2_tx_tnews.log &



CUDA_VISIBLE_DEVICES="1"  python3 finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/iflytek_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/iflytek_public/dev.json  \
                                        --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list  3 5 8   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > search_2_tx_ifly.log &


CUDA_VISIBLE_DEVICES="1"  python3 finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/ocnli_public/train.50k.json --dev_path /root/.paddlenlp/datasets/Clue/ocnli_public/dev.json  \
                                        --learning_rate_list 3e-5  1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list  3 5 8  \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128  > search_2_tx_ocnli.log &



CUDA_VISIBLE_DEVICES="2"  python3 finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/csl_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/csl_public/dev.json  \
                                        --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list  3 5 8  \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > search_2_tx_csl.log &


CUDA_VISIBLE_DEVICES="2" python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/cmnli_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/cmnli_public/dev.json  \
                                        --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list  3 5 8   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > search_2_tx_cmnli.log &

CUDA_VISIBLE_DEVICES="3"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/Clue/cluewsc2020_public/train.json --dev_path /root/.paddlenlp/datasets/Clue/cluewsc2020_public/dev.json  \
                                        --learning_rate_list 3e-5  1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list  3 5 8  \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > search_2_tx_wsc.log &

CUDA_VISIBLE_DEVICES="3"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/XNLI/xnli/train/part-0 --dev_path /root/.paddlenlp/datasets/XNLI/xnli/dev/part-0 \
                                        --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list  3 5 8   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > search_2_tx_xnli.log &

CUDA_VISIBLE_DEVICES="4"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/ChnSentiCorp/ChnSentiCorp/ChnSentiCorp/train.tsv --dev_path /root/.paddlenlp/datasets/ChnSentiCorp/ChnSentiCorp/ChnSentiCorp/dev.tsv \
                                        --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list 32 64 --epochs_num_list  3 5 8   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > search_2_tx_chnsenticorp.log &

CUDA_VISIBLE_DEVICES="4"  python3 -u finetune/run_classifier_grid.py --pretrained_model_path  /liujiaqi/pretrained_models/tencent/2l128d/pytorch_model.bin  \
                                        --vocab_path /liujiaqi/pretrained_models/tencent/2l128d/vocab.txt --config_path /liujiaqi/pretrained_models/tencent/2l128d/config.json \
                                        --train_path /root/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/train.tsv --dev_path /root/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/dev.tsv \
                                        --learning_rate_list 3e-5 1e-4 3e-4 --batch_size_list  32 64 --epochs_num_list  3 5 8   \
                                        --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 > search_2_tx_lcqmc.log &
