# demonstrates how to call the server and the client
# modify according to your needs

mkdir results
lambda=('0.8' '0.82' '0.84' '0.86' '0.88' '0.90' '0.92' '0.94' '0.96' '0.98' '1.0');
for lamb in "${lambda[@]}"
do   
    	for((n=0;n<10;n++))
	do
	    echo "----------------    SARSA 0 $n    ------------------"
	    python3 ./server/server.py -port $((5000+$n)) -i 1 -rs $n -ne 500 -q | tee "results/sarsa1t_accum_${lamb}_rs$n.txt" &
	    sleep 1
	    python3 ./client/client.py -port $((5000+$n)) -rs $n -gamma 1 -algo sarsa -lambda ${lamb}
	done
    
done



