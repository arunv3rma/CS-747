#!/usr/bin/python3.4

from collections import namedtuple
import argparse
import socket
import sys
# from agent import RandomAgent
import numpy as np

# python3.4 assignment3/client/client.py -ip 127.0.0.1 -port 5002 -algo sarsa -gamma 1 -lambda 0.1 -trace accum -rs 0

parser = argparse.ArgumentParser(description="Implements the Learning Agent.")
parser.add_argument('-ip', '--ip', dest='ip', type=str, default='localhost', help='IP of server')
parser.add_argument('-port', '--port', dest='port', type=int, default=5000, help='Port for connection')
parser.add_argument('-algo', '--algorithm', dest='algorithm', type=str, default='sarsa',
                    help='The learning algorithm to be used. {random, sarsa, qlearning, model}')
parser.add_argument('-gamma', '--gamma', dest='gamma', type=float, default=1, help='Discount Factor')
parser.add_argument('-lambda', '--lambda', dest='lamb', type=float, default=0, help='Value of lambda')
parser.add_argument('-trace', '--trace', dest='trace', type=str, default='accum',
                    help='Value of trace {accum, replace}')
parser.add_argument('-rs', '--randomseed', dest='randomseed', type=int, default=0, help='Seed for RNG.')
args = parser.parse_args()


def getResponse(message):
    global sock
    sock.sendall(message.encode())
    data = ''
    while True:
        data += sock.recv(1024).decode()
        if data[-1] == '\n':
            break
    if 'TERMINATE' in data:
        sys.exit()
    return data[:-1]


def epsilon_greedy(q_value, e, num_actions):
    e_step = np.random.binomial(1, e)
    if e_step == 1:
        selected_action = int(np.random.uniform(0, num_actions, 1))
    else:
        selected_action = np.argmax(q_value)
    return selected_action


def get_max(q_value):
    return np.argmax(q_value)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
Address = namedtuple('Address', ['ip', 'port'])

server_address = Address(ip=args.ip, port=args.port)
print('Connecting to:', server_address, file=sys.stderr)
sock.connect(server_address)

# ################## Code Start Here #############################
actions = ['up', 'down', 'left', 'right']  # up down left right
numActions = len(actions)
epsilon = 0.01
alpha = 0.1

try:
    print('Requesting environment info')
    numStates, state = map(int, getResponse('info').strip().split())
    print('Number of States: {}, Current State: {}\n=========='.format(numStates, state))

    if args.algorithm.lower() == 'qlearning':
        # print('Q')
        Q_val = np.random.rand(numStates, numActions)
        while True:
            action = epsilon_greedy(Q_val[state], epsilon, numActions)
            next_state, reward, event = map(int, getResponse(actions[action]).strip().split())
            best_action = np.argmax(Q_val[next_state])
            t_diff = reward + float(args.gamma) * Q_val[next_state][best_action] - Q_val[state][action]
            Q_val[state][action] += alpha * t_diff
            state = next_state

    elif args.algorithm.lower() == 'sarsa':
        Q_val = np.random.rand(numStates, numActions)
        e_val = np.zeros((numStates, numActions))
        action = epsilon_greedy(Q_val[state], epsilon, numActions)

        while True:
            pre_state = state
            state, reward, event = map(int, getResponse(actions[action]).strip().split())
            pre_action = action
            action = epsilon_greedy(Q_val[state], epsilon, numActions)

            delta = reward + float(args.gamma) * Q_val[state][action] - Q_val[pre_state][pre_action]

            if args.trace == 'accum':
                e_val[pre_state][pre_action] += 1
            else:
                e_val[pre_state][pre_action] = 1

            Q_val[pre_state][pre_action] += alpha * delta * e_val[pre_state][pre_action]
            e_val[pre_state][pre_action] *= float(args.gamma) * float(args.lamb)

            # This code is commented because taking time to process and not giving better performance
            # for s in range(numStates):
            #     for a in range(numActions):
            #         Q_val[s][a] += (1.0/t) * delta * e_val[s][a]  # alpha = 1/t
            #         e_val[s][a] *= float(args.gamma) * float(args.lamb)

            # event = 'continue terminated goal'.split()[event]
            # if event != 'continue':
            #     e_val = np.zeros((numStates, numActions))

            if event != 0:
                e_val = np.zeros((numStates, numActions))

    else:
        print('Invalid Algorithm, Try again with sarsa or qlearning')
        print('Closing Socket')
        sock.close()
        exit()
    #     agent = RandomAgent()
    #     # agent = Agent(numStates, state, args.gamma, args.lamb, args.algorithm.lower(), args.randomseed)
    #     while True:
    #         action = agent.getAction()  # Take action
    #         state, reward, event = map(int, getResponse(action).strip().split())
    #         event = 'continue terminated goal'.split()[event]
    #         agent.observe(state, reward, event)  # Observe Reward

# ################## Code End Here #############################

finally:
    print('Closing Socket')
    sock.close()
