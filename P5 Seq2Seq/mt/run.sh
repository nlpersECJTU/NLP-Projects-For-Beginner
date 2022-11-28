python3 ~/nlptasks/mt/train.py --config config/multi30k_en_de.json --batch_size 32 --use_beam 0 --seed 1234
sleep 5m

python3 ~/nlptasks/mt/train.py --config config/multi30k_en_de.json --batch_size 64 --use_beam 0 --seed 1234
sleep 5m

python3 ~/nlptasks/mt/train.py --config config/multi30k_en_de.json --batch_size 128 --use_beam 0 --seed 1234
sleep 5m

python3 ~/nlptasks/mt/train.py --config config/multi30k_en_de.json --batch_size 192 --use_beam 0 --seed 1234
sleep 5m

python3 ~/nlptasks/mt/train.py --config config/multi30k_en_de.json --batch_size 256 --use_beam 0 --seed 1234
sleep 5m

python3 ~/nlptasks/mt/train.py --config config/multi30k_en_de.json --batch_size 32 --use_beam 1 --seed 1234
sleep 5m

python3 ~/nlptasks/mt/train.py --config config/multi30k_en_de.json --batch_size 64 --use_beam 1 --seed 1234
sleep 5m

python3 ~/nlptasks/mt/train.py --config config/multi30k_en_de.json --batch_size 96 --use_beam 1 --seed 1234
sleep 5m
