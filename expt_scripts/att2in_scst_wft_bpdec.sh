#!/bin/bash -eu

cd $HOME/switch_disc_caption
export PYTHONPATH=$PYTHONPATH:`pwd`

ID=att2in_scst_wft_bpdec
LR=1e-4
TMP=1

# fine-tune
python -u tools/fine_tune.py --id ${ID} --id_old a2i2_sc-best --caption_model poe_a2i2 --max_epochs 1 --batch_size 10 --learning_rate ${LR} --start_from saved_models/att2in_scst --checkpoint_path saved_models/${ID} --poe_temp ${TMP}

# evaluate on val
python -u tools/eval.py --split val --dump_images 0 --num_images 5000 --model saved_models/${ID}/model.pth --infos_path saved_models/${ID}/infos_${ID}.pkl --language_eval 1 --input_fc_dir data/cocotalk_fc --beam_size 5 --poe_decode ${TMP}

# evaluate on test
python -u tools/eval.py --split test --dump_images 0 --num_images 5000 --model saved_models/${ID}/model.pth --infos_path saved_models/${ID}/infos_${ID}.pkl --language_eval 1 --input_fc_dir data/cocotalk_fc --beam_size 5 --poe_decode ${TMP}

echo Done