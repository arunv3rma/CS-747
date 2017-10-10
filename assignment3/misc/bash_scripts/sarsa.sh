# demonstrates how to call the server and the client
# modify according to your needs

mkdir results

lambda=('0' '0.1' '0.15' '0.2' '0.25' '0.3' '0.35' '0.4' '0.5' '0.6' '0.65' '0.7' '0.8' '0.85' '0.9' '1.0');
for lamb in "${lambda[@]}"
do   
    	for((n=0;n<10;n++))
	do
	    echo "----------------    SARSA 0 $n    ------------------"
	    python3 ./server/server.py -port $((5000+$n)) -i 0 -rs $n -ne 500 -q | tee "results/sarsa_accum_${lamb}_rs$n.txt" &
	    sleep 1
	    python3 ./client/client.py -port $((5000+$n)) -rs $n -gamma 1 -algo sarsa -lambda ${lamb}
	done
    
done



