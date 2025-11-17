@echo off
REM Stage 1: DemonAttack (0-20M)
python -m scripts.train --algo ppo --env ALE/DemonAttack-v5 --model demon_carnival_name_baseline --frames 20000000 --procs 8 --save-interval 10

REM Stage 2: Carnival (20M-30M)
python -m scripts.train --algo ppo --env ALE/Carnival-v5 --model demon_carnival_name_baseline --frames 30000000 --procs 8 --save-interval 10

REM Stage 3: NameThisGame (30M-40M)
python -m scripts.train --algo ppo --env ALE/NameThisGame-v5 --model demon_carnival_name_baseline --frames 40000000 --procs 8 --save-interval 10