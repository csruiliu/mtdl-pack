#datetime=$(date +%Y%m%d%H%M)
#timestamp=$(date +%Y%m%d%H%M%S.%3N)

#Query Type can be one of the following:
#utilization.gpu
#utilization.memory
#memory.total
#memory.free
#memory.used


PYCMD=""
NVISMI=""
lms=10
#queryType="memory.used,memory.free,memory.total,utilization.memory,utilization.gpu"
queryType="memory.used"
while getopts "f:" opt; do
    case $opt in
        f)
        fileName=$OPTARG
        ;;
    esac
done

#FilePath=/home/ruiliu/Development/mtml-tf/pack-exp/$fileName
#FilePath=/tank/local/ruiliu/mtml-tf/pack-exp/$fileName
FilePath=/home/user/Development/mtml-tf/pack-exp/$fileName

if [ -f $FilePath ]
then
    echo "log file exists, removing it"
    rm $FilePath
fi

#echo $PYCMD
python3 pack_train_profiler.py &
nvidia-smi --query-gpu=$queryType --format=csv --loop-ms=$lms >> $FilePath &
wait -n
sleep 5
pkill -P $$
