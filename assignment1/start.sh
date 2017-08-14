#!/bin/bash

PWD=`pwd`
SERVERDIR=./server
CLIENTDIR=./client

horizon=4000


hostname="localhost"
port=5001

#base_port=5001
banditFile="$PWD/data/instance-25.txt"
epsilon=0.15

#Horizons=('10' '100' '1000' '10000' '100000');
Horizons=('100');
#Algos=('epsilon-greedy' 'UCB' 'Thompson-Sampling' 'KL-UCB');
Algos=('epsilon-greedy');
#Algos=('Thompson-Sampling')

nRuns=10
numArms=$(wc -l $banditFile | cut -d" " -f1 | xargs)
#echo $numArms
#port_count=0
for algorithm in "${Algos[@]}"
do

    for horizon in "${Horizons[@]}"
    do

        for ((i=1; i<=${nRuns}; i++))
        do
#            port_count=$[$port_count +1]
#            port=$(($base_port + $port_count))
            PIDS=$(ps -ef | grep "bandit-environment" | head -n -1 | awk '{print $2}')
            echo $PIDS
            if [ -z "$PIDS" ]; then
                randomSeed=0
                OUTPUTFILE="$PWD/${algorithm}_${horizon}_$i.txt"
    #            echo $OUTPUTFILE

                pushd $SERVERDIR
                cmd="./startserver.sh $numArms ${horizon} $port $banditFile $randomSeed $OUTPUTFILE &"
    #            echo $cmd
                $cmd
                popd
                sleep 2

                pushd $CLIENTDIR
                cmd="./startclient.sh $numArms $horizon $hostname $port $randomSeed $algorithm $epsilon &"
                echo $cmd
                $cmd /dev/null
                sleep 1
                popd
            else
                sleep 1
            fi

        done

    done

done
