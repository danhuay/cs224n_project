## single task, simple concat
#python main.py --name baseline_single_task_sst_bert_sep_concat \
#  --params train_data_tags=sst, \
#  --params training_approach=sequential

#python main.py --name ext1_tsa_para_exp \
#  --params train_data_tags=para, \
#  --params training_approach=sequential \
#  --params epochs=50 \
#  --params tsa_schedule=exp
#
for schedule in back_translation completion rnd_mask_completion
do
 python main.py --name ddd_uda_800_simul_linear_$schedule \
   --params aug_approach=$schedule
done

# schedule=log
# python main.py --name ext1_tsa_all_simul_$schedule \
#   --params training_approach=sequential \
#   --params epochs=50 \
#   --params tsa_schedule=$schedule

#python main.py --name ext1_tsa_all_simul_log \
#  --params training_approach=simultaneous \
#  --params epochs=25 \
#  --params tsa_schedule=log
#
#python main.py --name ext1_tsa_all_simul_linear \
#  --params training_approach=simultaneous \
#  --params epochs=25 \
#  --params tsa_schedule=linear



#python main.py --name baseline_single_task_sts_bert_sep_concat \
#  --params train_data_tags=sts, \
#  --params training_approach=sequential

## multi-tasks, simple concat
#for batch_size in 8
#do
#  # sequential
#  python main.py --name baseline_multi_task_sequential_para_bert_sep_concat \
#    --params train_data_tags=para,sts,sst \
#    --params training_approach=sequential \
#    --params batch_size=$batch_size
#
#  # simultaneous
#  python main.py --name baseline_multi_task_simul_para_bert_sep_concat  \
#    --params train_data_tags=para,sts,sst \
#    --params training_approach=simultaneous \
#    --params batch_size=$batch_size
#done
