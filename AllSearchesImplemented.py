import sys
from collections import deque
import heapq
from typing import List, Tuple
from queue import PriorityQueue
import queue
from datetime import datetime
def read_input_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    grid = []
    for line in lines:
        if line.strip() == "END OF FILE":
            break
        grid.append([int(x) for x in line.strip().split()])
    return grid
def write_output_file(filename, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, solution_depth, solution_cost, steps):
    with open(filename, "w") as f:
        f.write("Nodes Popped: {}\n".format(nodes_popped))
        f.write("Nodes Expanded: {}\n".format(nodes_expanded))
        f.write("Nodes Generated: {}\n".format(nodes_generated))
        f.write("Max Fringe Size: {}\n".format(max_fringe_size))
        f.write("Solution Found at depth {} with cost of {}.\n".format(solution_depth, solution_cost))
        f.write("Steps:\n")
        for step in steps:
            f.write("\t{}\n".format(step))



def greedy(start_state, goal_state, dump_flag):
    now=datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    trace_file=None
    if dump_flag:
        trace_file=open('%s.txt' % dt_string,'w')
    moves = ['Up', 'Left', 'Down', 'Right']
    cost = 0
    popped = 0
    expanded = 0
    generated = 0
    max_fringe = 0
    depth = 0
    steps = []
    fringe = PriorityQueue()
    fringe.put((0, start_state, 0, steps))

    while not fringe.empty():
        current = fringe.get()
        
        popped += 1
        if current[1] == goal_state:
            cost = current[2]
            depth = len(current[3])
            steps = current[3]
            break
        if dump_flag:
            trace_file.write(str(current[1]))
            trace_file.write(f"\nGenerating Successors cost: {cost} depth: {depth} steps: {steps}\n")
        if len(fringe.queue) > max_fringe:
            max_fringe = len(fringe.queue)

        expanded += 1
        for move in moves:
            if move == 'Up':
                next_state = move_up(current[1])
                if next_state:
                    generated += 1
                    i,j=find_blank(current[1])
                    steps = current[3] + [f'Move {current[1][i-1][j]} Down']
                    fringe.put((get_heuristic(next_state, goal), next_state, current[2] + current[1][i-1][j], steps))
                    if dump_flag:
                        trace_file.write(str(current[1]))
                        trace_file.write(f"\nSuccessor: {current[1][i][j]} move {current[1][i-1][j]} down cost: {current[2] + current[1][i-1][j]}\n")

            elif move == 'Left':
                next_state = move_left(current[1])
                if next_state:
                    generated += 1
                    i,j=find_blank(current[1])
                    steps = current[3] + [f'Move {current[1][i][j-1]} Right']
                    fringe.put((get_heuristic(next_state, goal), next_state, current[2] + current[1][i][j-1], steps))
                    if dump_flag:
                        trace_file.write(str(current[1]))
                        trace_file.write(f"\nSuccessor:  move {current[1][i][j-1]} down cost: {current[2] + current[1][i][j-1]}\n")

            elif move == 'Down':
                next_state = move_down(current[1])
                if next_state:
                    generated += 1
                    i,j=find_blank(current[1])
                    steps = current[3] + [f'Move {current[1][i+1][j]} Up']
                    fringe.put((get_heuristic(next_state, goal), next_state, current[2] + current[1][i+1][j], steps))
                    if dump_flag:
                        trace_file.write(str(current[1]))
                        trace_file.write(f"\nSuccessor: {current[1][i][j]} move {current[1][i+1][j]} down cost: {current[2] + current[1][i+1][j]}\n")

            elif move == 'Right':
                next_state = move_right(current[1])
                if next_state:
                    generated += 1
                    i,j=find_blank(current[1])
                    steps = current[3] + [f'Move {current[1][i][j+1]} Left']
                    fringe.put((get_heuristic(next_state, goal), next_state, current[2] + current[1][i][j+1], steps))
                    if dump_flag:
                        trace_file.write(str(current[1]))
                        trace_file.write(f"\nSuccessor: {current[1][i][j]} move {current[1][i][j+1]} down cost: {current[2] + current[1][i][j+1]}\n")
    print("Nodes Popped: ",popped)
    print("Nodes Expanded: ",expanded)
    print("Nodes Generated: ",generated)
    print("Max Fringe Size: ", max_fringe)
    print("Solution Found at depth: "+str(depth)+" with cost of: "+str(cost))
    print("Steps:\n")
    for step in steps:
        print(step)


def a_star(start, goal, dump_flag):
    now=datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    trace_file=None
    if dump_flag:
        trace_file=open('%s.txt' % dt_string,'w')
    steps = []
    cost = 0
    max_fringe_size = 0
    nodes_generated = 0
    nodes_expanded = 0
    nodes_popped = 0
    
    visited = set()
    heap = []
    heapq.heappush(heap, (0, cost, start, steps))
    
    while heap:
        current_cost, current_cost_val, current_state, current_steps = heapq.heappop(heap)
        if dump_flag:
            trace_file.write(str(current_state))
            trace_file.write(f"\nGenerating Successors: cost {current_cost} current steps: {current_steps}\n")
        nodes_popped += 1
        if current_state == goal:
            return {
                "Nodes Popped": nodes_popped,
                "Nodes Expanded": nodes_expanded,
                "Nodes Generated": nodes_generated,
                "Max Fringe Size": max_fringe_size,
                "Solution Found at depth": len(current_steps),
                "cost": current_cost_val,
                "Steps": current_steps
            }
        
        if tuple(map(tuple, current_state)) in visited:
            continue
        
        visited.add(tuple(map(tuple, current_state)))
        nodes_expanded += 1
        
        for move_func, direction in zip([move_up, move_left, move_down, move_right], ["Up", "Left", "Down", "Right"]):
            new_state = move_func(current_state)
            if new_state:
                nodes_generated += 1
                new_cost = current_cost_val + current_state[find_blank(new_state)[0]][find_blank(new_state)[1]]
                if direction=="Up":
                    direction="Down"
                if direction=="Down":
                    direction="Up"
                if direction=="Left":
                    direction="Right"
                if direction=="Right":
                    direction="Left"
                new_steps = current_steps[:] + [f"Move {current_state[find_blank(new_state)[0]][find_blank(new_state)[1]]} {direction}"]
                heapq.heappush(heap, (new_cost + get_heuristic(new_state, goal), new_cost, new_state, new_steps))
                if dump_flag:
                    trace_file.write(f"Successor: {new_state} steps: {new_steps} cost: {new_cost}\n")
                max_fringe_size = max(max_fringe_size, len(heap))
    
    return None



def bfs(start_state, goal_state, dump_flag):
    now=datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    trace_file=None
    if dump_flag:
        trace_file=open('%s.txt' % dt_string,'w')
    nodes_generated = 0
    nodes_expanded = 0
    max_fringe_size = 0
    
    fringe = deque([(start_state, [], 0)])
    visited = list()
    
    while fringe:
        node, path, cost = fringe.popleft()
        nodes_generated += 1
        if dump_flag:
            trace_file.write(f"Generating successors to Node: {node} Path: {path} Cost: {cost}\n")
        if node == goal_state:
            return {
                "Nodes Popped": nodes_generated,
                "Nodes Expanded": nodes_expanded,
                "Nodes Generated": nodes_generated,
                "Max Fringe Size": max_fringe_size,
                "Solution Found at depth": len(path),
                "cost": cost,
                "Steps": path
            }
        
        if tuple(node) not in visited:
            visited.append(tuple(node))
            nodes_expanded += 1
            
            up = move_up(node)
            if up:
                i,j=find_blank(up)
                i+=1
                fringe.append((up, path + ["Move " + str(up[i][j]) + " Down"], cost + up[i][j]))
                if dump_flag:
                    trace_file.write("Successor: "+str(fringe))
            left = move_left(node)
            if left:
                i,j=find_blank(left)
                j+=1
                fringe.append((left, path + ["Move " + str(left[i][j]) + " Right"], cost + left[i][j]))
                if dump_flag:
                    trace_file.write("Successor: "+str(fringe))
            
            down = move_down(node)
            if down:
                i,j=find_blank(down)
                i-=1
                fringe.append((down, path + ["Move " + str(down[i][j]) + " Up"], cost + down[i][j]))
                if dump_flag:
                    trace_file.write("Successor: "+str(fringe))
            
            right = move_right(node)
            if right:
                i,j=find_blank(right)
                j-=1
                fringe.append((right, path + ["Move " + str(right[i][j]) + " Left"], cost + right[i][j]))
                if dump_flag:
                    trace_file.write("Successor: "+str(fringe))
            
            max_fringe_size = max(max_fringe_size, len(fringe))
    
    return None


def ucs(start_state, goal_state, dump_flag):
    now=datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    trace_file=None
    if dump_flag:
        trace_file=open('%s.txt' % dt_string,'w')        
    q = PriorityQueue()
    q.put((0, start_state, [], 0))
    nodes_generated = 1
    nodes_expanded = 0
    nodes_popped = 0
    max_fringe_size = 1
    while not q.empty():
        cost, state, actions, depth = q.get()
        if dump_flag:
            trace_file.write(f"Generating Successors: {state} cost: {cost} steps: {actions}\n")
        nodes_popped += 1
        if state == goal_state:
            return (
                "Nodes Popped: " + str(nodes_popped) + "\n" +
                "Nodes Expanded: " + str(nodes_expanded) + "\n" +
                "Nodes Generated: " + str(nodes_generated) + "\n" +
                "Max Fringe Size: " + str(max_fringe_size) + "\n" +
                "Solution Found at depth " + str(depth) + " with cost of " + str(cost) + ".\n" +
                "Steps:\n" + "\n".join(actions)
            )
        up = move_up(state)

        if up:
            i,j=find_blank(up)
            i+=1
            q.put((cost + up[i][j], up, actions + ["Move " + str(up[i][j]) + " Down"], depth + 1))
            if dump_flag:
                trace_file.write(f"Successor: {up[i][j]} cost: {cost} steps: {actions}\n")
            nodes_generated += 1
            max_fringe_size = max(max_fringe_size, q.qsize() + 1)
        left = move_left(state)
        if left:
            i,j=find_blank(left)
            j+=1
            q.put((cost + left[i][j], left, actions + ["Move " + str(left[i][j]) + " Right"], depth + 1))
            if dump_flag:
                trace_file.write(f"Successor: {left[i][j]} cost: {cost} steps: {actions}\n")
            nodes_generated += 1
            max_fringe_size = max(max_fringe_size, q.qsize() + 1)
        down = move_down(state)
        if down:
            i,j=find_blank(down)
            i-=1
            q.put((cost + down[i][j], down, actions + ["Move " + str(down[i][j]) + " Up"], depth + 1))
            if dump_flag:
                trace_file.write(f"Successor: {down[i][j]} cost: {cost} steps: {actions}\n")
            nodes_generated += 1
            max_fringe_size = max(max_fringe_size, q.qsize() + 1)
        right = move_right(state)
        if right:
            i,j=find_blank(right)
            j-=1
            q.put((cost + right[i][j], right, actions + ["Move " + str(right[i][j]) + " Left"], depth + 1))
            if dump_flag:
                trace_file.write(f"Successor: {right[i][j]} cost: {cost} steps: {actions}\n")
            nodes_generated += 1
            max_fringe_size = max(max_fringe_size, q.qsize() + 1)
        nodes_expanded += 1
    return "No solution found."



def dfs(start, goal, dump_flag):    
    now=datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    trace_file=None
    if dump_flag:
        trace_file=open('%s.txt' % dt_string,'w')    
    visited = set()
    stack = [(start, 0, [],0)]
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 0
    
    while stack:
        nodes_popped += 1
        state, depth, moves,cost = stack.pop()
        if dump_flag:
            trace_file.write(f"Generating Successors: {state} cost: {cost} steps: {moves}\n")
        if state == goal:
            return (
                "Nodes Popped: " + str(nodes_popped) + "\n" +
                "Nodes Expanded: " + str(nodes_expanded) + "\n" +
                "Nodes Generated: " + str(nodes_generated) + "\n" +
                "Max Fringe Size: " + str(max_fringe_size) + "\n" +
                "Solution Found at depth " + str(depth) + " with cost of " + str(cost) + ".\n" +
                "Steps:\n" + "\n".join("\t" + move for move in moves[::-1])
            )
        
       
        visited.add(str(state))
        
        nodes_expanded += 1
        up = move_up(state)
        if up and str(up) not in visited:
            i,j=find_blank(up)
            i+=1
            stack.append((up, depth + 1, moves + ["Move " + str(up[i][j]) + " Down"],cost+up[i][j]))
            if dump_flag:
                trace_file.write(f"Successor: {up} cost: {cost+up[i][j]} steps: {moves}\n")
            nodes_generated += 1
        
        left = move_left(state)
        if left and str(left) not in visited:
            i,j=find_blank(left)
            j+=1
            stack.append((left, depth + 1, moves + ["Move " + str(left[i][j]) + " Right"],cost+left[i][j]))
            if dump_flag:
                trace_file.write(f"Successor: {left} cost: {cost+left[i][j]} steps: {moves}\n")
            nodes_generated += 1
        
        down = move_down(state)
        if down and str(down) not in visited:
            i,j=find_blank(down)
            i-=1
            stack.append((down, depth + 1, moves + ["Move " + str(down[i][j]) + " Up"],cost+down[i][j]))
            if dump_flag:
                trace_file.write(f"Successor: {down} cost: {cost+down[i][j]} steps: {moves}\n")
            nodes_generated += 1
        
        right = move_right(state)
        if right and str(right) not in visited:
            i,j=find_blank(right)
            j-=1
            stack.append((right, depth + 1, moves + ["Move " + str(right[i][j]) + " Left"],cost+right[i][j]))
            if dump_flag:
                trace_file.write(f"Successor: {right} cost: {cost+right[i][j]} steps: {moves}\n")
            nodes_generated += 1
        
        max_fringe_size = max(max_fringe_size, len(stack))
    
    return "No solution found"


def dls(start, goal, limit, dump_flag):
    now=datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    trace_file=None
    if dump_flag:
        trace_file=open('%s.txt' % dt_string,'w')
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 0
    max_fringe_size = 0
    fringe = [(start, 0, [],0)]
    while fringe:
        max_fringe_size = max(max_fringe_size, len(fringe))
        node, depth, moves,cost = fringe.pop(0)
        if dump_flag:
            trace_file.write(f"Generating Successors: {node} cost: {cost} steps: {moves}\n")
        nodes_popped += 1
        if node == goal:
            return (
                "Nodes Popped: " + str(nodes_popped) +
                "\nNodes Expanded: " + str(nodes_expanded) +
                "\nNodes Generated: " + str(nodes_generated) +
                "\nMax Fringe Size: " + str(max_fringe_size) +
                "\nSolution Found at depth " + str(depth) + " with cost of " + str(cost) + "." +
                "\nSteps:\n\t" + "\n\t".join(moves)
            )
        if depth < limit:
            nodes_expanded += 1
            up = move_up(node)
            if up:
                nodes_generated += 1
                i,j=find_blank(up)
                i+=1
                fringe.append((up, depth + 1, moves + ["Move " + str(up[i][j]) + " Down"],cost+up[i][j]))
                if dump_flag:
                    trace_file.write(f"Successor: {up} cost: {cost+up[i][j]} steps: {moves}\n")
            left = move_left(node)
            if left:
                i,j=find_blank(left)
                j+=1
                nodes_generated += 1
                fringe.append((left, depth + 1, moves + ["Move " + str(left[i][j]) + " Right"],cost+left[i][j]))
                if dump_flag:
                    trace_file.write(f"Successor: {left} cost: {cost+left[i][j]} steps: {moves}\n")
            down = move_down(node)
            if down:
                i,j=find_blank(down)
                nodes_generated += 1
                i-=1
                fringe.append((down, depth + 1, moves + ["Move " + str(down[i][j]) + " Up"],cost+down[i][j]))
                if dump_flag:
                    trace_file.write(f"Successor: {down} cost: {cost+down[i][j]} steps: {moves}\n")
            right = move_right(node)
            if right:
                i,j=find_blank(right)
                j-=1
                nodes_generated += 1
                fringe.append((right, depth + 1, moves + ["Move " + str(right[i][j]) + " Left"],cost+right[i][j]))
                if dump_flag:
                    trace_file.write(f"Successor: {right} cost: {cost+right[i][j]} steps: {moves}\n")
    return "Solution Not Found"
         
           
def ids(start, goal, dump_flag):
    max_depth = 0
    while True:
        result = dls(start, goal, max_depth,dump_flag)
        if result!="Solution Not Found":
            return result
        max_depth += 1




def get_heuristic(tile, goal):
    h = 0
    for i in range(3):
        for j in range(3):
            if tile[i][j] != goal[i][j]:
                h += 1
    return h
def move_up(state):
    blank_row, blank_col = find_blank(state)
    if blank_row > 0:
        new_state = [row[:] for row in state]
        new_state[blank_row][blank_col], new_state[blank_row-1][blank_col] = new_state[blank_row-1][blank_col], new_state[blank_row][blank_col]
        return new_state
    return None
def move_left(state):
    blank_row, blank_col = find_blank(state)
    if blank_col > 0:
        new_state = [row[:] for row in state]
        new_state[blank_row][blank_col], new_state[blank_row][blank_col-1] = new_state[blank_row][blank_col-1], new_state[blank_row][blank_col]
        return new_state
    return None
def move_down(state):
    blank_row, blank_col = find_blank(state)
    if blank_row < 2:
        new_state = [row[:] for row in state]
        new_state[blank_row][blank_col], new_state[blank_row+1][blank_col] = new_state[blank_row+1][blank_col], new_state[blank_row][blank_col]
        return new_state
    return None
def move_right(state):
    blank_row, blank_col = find_blank(state)
    if blank_col < 2:
        new_state = [row[:] for row in state]
        new_state[blank_row][blank_col], new_state[blank_row][blank_col+1] = new_state[blank_row][blank_col+1], new_state[blank_row][blank_col]
        return new_state
    return None
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return (i, j)
    return None




if __name__ == "__main__":
    start_filename = sys.argv[1]
    goal_filename = sys.argv[2]
    method = "a_star"
    if len(sys.argv) > 3:
        method = sys.argv[3]
    dump_flag = False
    if len(sys.argv) > 4:
        dump_flag = sys.argv[4] == "true"

    start = read_input_file(start_filename)
    goal = read_input_file(goal_filename)

    if method == "bfs":
        solution=bfs(start, goal, dump_flag)
        print("Nodes Popped: ",solution['Nodes Popped'])
        print("Nodes Expanded: ",solution['Nodes Expanded'])
        print("Nodes Generated: ",solution['Nodes Generated'])
        print("Max Fringe Size: ",solution['Max Fringe Size'])
        print("Solution Found at depth: ",solution['Solution Found at depth'])
        print("Cost: ",solution['cost'])
        print("Steps:\n")
        for step in solution['Steps']:
            print(step)
    elif method == "ucs":
        solution=ucs(start,goal,dump_flag)
        print(solution)
        
    elif method == "dfs":
         solution=dfs(start, goal, dump_flag)
         print(solution)
        #  print("Steps: ")
        #  for step in steps:
        #     print(step)
    elif method == "dls":
        max_depth=int(input("Enter max depth: "))
        solution= dls(start, goal,max_depth, dump_flag)
        print(solution)
    elif method == "ids":
         solution = ids(start, goal, dump_flag)
         print(solution)
    elif method == "greedy":
        solution=greedy(start,goal,dump_flag)
       
    elif method == "a*":
         solution = a_star(start, goal, dump_flag)
         print("Nodes Popped: ",solution['Nodes Popped'])
         print("Nodes Expanded: ",solution['Nodes Expanded'])
         print("Nodes Generated: ",solution['Nodes Generated'])
         print("Max Fringe Size: ",solution['Max Fringe Size'])
         print("Solution Found at depth: ",solution['Solution Found at depth'])
         print("Cost: ",solution['cost'])
         print("Steps:\n")
         for step in solution['Steps']:
            print(step)
