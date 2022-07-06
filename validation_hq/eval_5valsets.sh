DATAROOT=/share/team/thucth/data/FaceReg
PRETRAINED=experiments/run_ir101_ms1mv2_full_finetune_06-27_0/epoch=25-step=1084483.ckpt

python main.py \
    --data_root ${DATAROOT} \
    --train_data_path faces_emore/imgs \
    --val_data_path faces_emore \
    --prefix adaface_ir101_ms1mv2 \
    --gpus 2\
    --use_16bit \
    --start_from_model_statedict ${PRETRAINED} \
    --arch ir_101 \
    --evaluate
