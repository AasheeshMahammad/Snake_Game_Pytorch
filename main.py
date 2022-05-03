import torch,random
import numpy as np
from game import Snake_Game_Agent
from collections import deque
from model import Linear_QNet,QTrainer
from plotter import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = .001  # learning rate

useGpu = True
if torch.cuda.is_available() and useGpu:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class Agent:

    def __init__(self) -> None:
        self.n_games=0
        self.epsilon=0
        self.gamma=0.9
        self.memory=deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11,256,3,DEVICE)
        try:
            self.model.load_state_dict(torch.load("model/model.pth"))
        except FileNotFoundError:
            pass
        self.trainer = QTrainer(self.model,LR,self.gamma,DEVICE)
        

    def get_state(self,game):
        points = game.get_points()
        direction_state = game.get_state()
        #clock_wise=['RIGHT','DOWN','LEFT','UP']
        state = [
            #straight collision
            (direction_state['RIGHT'] and game.collides(points['RIGHT'])) or
            (direction_state['LEFT'] and game.collides(points['LEFT'])) or
            (direction_state['UP'] and game.collides(points['UP'])) or
            (direction_state['DOWN'] and game.collides(points['DOWN'])),
            #right collision
            (direction_state['RIGHT'] and game.collides(points['DOWN'])) or
            (direction_state['LEFT'] and game.collides(points['UP'])) or
            (direction_state['UP'] and game.collides(points['RIGHT'])) or
            (direction_state['DOWN'] and game.collides(points['LEFT'])),
            #left collision
            (direction_state['RIGHT'] and game.collides(points['UP'])) or
            (direction_state['LEFT'] and game.collides(points['DOWN'])) or
            (direction_state['UP'] and game.collides(points['LEFT'])) or
            (direction_state['DOWN'] and game.collides(points['RIGHT'])),

            #direction states
            direction_state['LEFT'],
            direction_state['RIGHT'],
            direction_state['UP'],
            direction_state['DOWN'],

            #food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state,dtype=int)



    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        state,action,reward,next_state,done = zip(*mini_sample)
        self.trainer.train_step(state,action,reward,next_state,done)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        self.epsilon = 20 - self.n_games
        final_action = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move_ind= random.randint(0,2)
            final_action[move_ind] = 1
        else:
            state_0 = torch.tensor(state,dtype=torch.float,device=DEVICE)
            prediction = self.model(state_0)
            move_ind = torch.argmax(prediction).item()
            final_action[move_ind] = 1
        return final_action

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Snake_Game_Agent()
    
    while True:
        old_state = agent.get_state(game)
        final_action = agent.get_action(old_state)
        reward,alive,score = game.play(final_action)
        new_state = agent.get_state(game)
        agent.train_short_memory(old_state,final_action,reward,new_state,alive)
        agent.remember(old_state,final_action,reward,new_state,alive)

        if not alive:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()

            print('Game :',agent.n_games,'Score :',score,'Record :',record)
            plot_scores.append(score)
            total_score +=score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

if __name__=='__main__':
    train()