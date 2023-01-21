export MCTS_TIME=0
export DO_LOWER_BOUND_RATIO_CONDITION=0
export ILP_OR_LP='ILP'

for file in graphs/sccs/*
do
 process_name="python3 ExactSolver.py $file"
 python3 ExactSolver.py $file & { sleep 1801; pkill -f -x $process_name;} &
 echo "$file"
done;
sleep 1802;
python3 sssend.py;