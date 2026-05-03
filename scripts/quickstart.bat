@echo off
python scripts\make_dataset.py --config configs\data\dataset.yaml --num-samples 8 --n-steps 60 --save-every 10
python src\data_gen\preprocess.py --config configs\data\preprocess.yaml
python scripts\train.py --config configs\model\fno.yaml
python scripts\eval_accuracy.py --config configs\model\fno.yaml --checkpoint experiments\fno\best.pt
echo Done.
