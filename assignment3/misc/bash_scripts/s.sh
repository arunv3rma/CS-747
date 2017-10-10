# demonstrates how to call the server and the client
# modify according to your needs

mkdir results

for((n=0;n<10;n++))
do
    echo "----------------    SARSA \0 $n    ------------------"
    python3 ./server/server.py -port $((5000+$n)) -i 0 -rs $n -ne 1500 -q | tee "results/sarsa_accum_rs$n.txt" &
    sleep 1
    python3 ./client/client.py -port $((5000+$n)) -rs $n -gamma 1 -algo sarsa -lambda 1.0
done

