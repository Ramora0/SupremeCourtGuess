sbatch slurms/a100.slurm 3b-baseline --model 3b
sbatch slurms/a100.slurm 3b-lr1e4 --model 3b --lr 1e-4
sbatch slurms/a100.slurm 3b-lr1e5 --model 3b --lr 1e-5
sbatch slurms/a100.slurm 3b-small-head --model 3b --head-dim 32 --num-queries 2 --self-attn-layers 1 --ffn-dim 128
sbatch slurms/a100.slurm 3b-large-head --model 3b --head-dim 128 --num-queries 8 --self-attn-layers 2 --ffn-dim 512
sbatch slurms/a100.slurm 3b-accum8 --model 3b --grad-accum 8
sbatch slurms/a100.slurm 3b-accum32 --model 3b --grad-accum 32
