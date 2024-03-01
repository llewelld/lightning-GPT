stdbuf -o0 xpu-smi dump -t 0,1 -m 0,1,2,5 -i 2 > "dmon-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${PMI_RANK}".txt
