set -e
ID="$1"
OUT="outputs/output${ID:+_$ID}.csv"

python src/train.py
python src/train.py --no-time --name no_time
python src/train.py --bools --name bools
python src/train.py --bools --present --category --name columns
python src/train.py --bools --present --category --no-time --name columns_no_time
python src/train.py --aggr mean --name mean_only
python src/train.py --layers 1 --name one_layer
python src/train.py --no-reverse --name no_reverse

python src/predict.py ${ID:+--id "$ID"}
python src/predict.py ${ID:+--id "$ID"} --name no_time
python src/predict.py ${ID:+--id "$ID"} --name bools
python src/predict.py ${ID:+--id "$ID"} --name columns
python src/predict.py ${ID:+--id "$ID"} --name columns_no_time
python src/predict.py ${ID:+--id "$ID"} --name mean_only
python src/predict.py ${ID:+--id "$ID"} --name one_layer
python src/predict.py ${ID:+--id "$ID"} --name no_reverse

python src/evaluate.py --pred "$OUT" --input test_input.csv