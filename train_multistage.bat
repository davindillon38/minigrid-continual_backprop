# Baseline: fresh Pong training
python -m scripts.train --algo ppo --env ALE/Pong-v5 --model pong_baseline --frames 5000000 --procs 8 --save-interval 10

# CB: fresh Pong training  
python -m scripts.train --algo ppo --env ALE/Pong-v5 --model pong_cb --frames 10000000 --procs 8 --save-interval 10 --continual-backprop