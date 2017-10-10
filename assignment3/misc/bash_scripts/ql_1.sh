# demonstrates how to call the server and the client
# modify according to your needs

mkdir results

for((n=0;n<10;n++))
do
    echo "----------------    Q Learning $n    ------------------"
    python3 ./server/server.py -port $((4000+$n)) -i 1 -rs $n -ne 1500 -q | tee "results/qlearning1_rs$n.txt" &
    sleep 1
    python3 ./client/client.py -port $((4000+$n)) -rs $n -gamma 1 -algo qlearning
done

