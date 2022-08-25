import random,os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from enum import Enum
from collections import namedtuple
import numpy as np

class Directions(Enum):
    RIGHT=1
    LEFT=2
    UP=3
    DOWN=4

class const:
    Point = namedtuple('Point',['x','y'])
    BLOCK_SIZE = 20
    SPEED = 100
    COLORS={"white":(255,255,255),"red":(200,0,0),"black":(0,0,0),
            "blue":(0,0,225),"green":(0,255,0),"brown":(165,42,42)}
    pygame.font.init()
    FONT=pygame.font.SysFont('arial',19)


class Snake_Game_Agent:
    def __init__(self,width=640,height=480) -> None:
        pygame.init()
        self.width=width
        self.height=height

        self.display=pygame.display.set_mode((self.width,self.height))
        pygame.display.set_caption('Snake Game')
        self.clock=pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction=Directions.UP
        self.head=const.Point(self.width/2,self.height/2)
        #print(self.head)
        self.snake=[self.head,const.Point(self.head.x-const.BLOCK_SIZE,self.head.y),
                    const.Point(self.head.x-(2*const.BLOCK_SIZE),self.head.y)]
        self.score=0
        self.food=None
        self._place_food()
        self.frame_iteration=0

    def get_points(self):
        points={
            'RIGHT': const.Point(self.head.x+const.BLOCK_SIZE,self.head.y),
            'LEFT': const.Point(self.head.x-const.BLOCK_SIZE,self.head.y),
            'UP': const.Point(self.head.x,self.head.y+const.BLOCK_SIZE),
            'DOWN': const.Point(self.head.x,self.head.y-const.BLOCK_SIZE)
        }
        return points
    
    def get_distance(self):
        distance = ((self.food.x-self.head.x)**2 + (self.food.y-self.head.y)**2)**0.5
        return distance

    def _place_food(self):
        x=random.randint(0,(self.width-const.BLOCK_SIZE)//const.BLOCK_SIZE)*const.BLOCK_SIZE
        y=random.randint(0,(self.height-const.BLOCK_SIZE)//const.BLOCK_SIZE)*const.BLOCK_SIZE
        self.food=const.Point(x,y)
        if self.food in self.snake:
            self._place_food()

    def get_state(self):
        direction_state = {'LEFT':self.direction == Directions.LEFT,
                 'RIGHT':self.direction == Directions.RIGHT,
                 'UP':self.direction == Directions.UP,
                 'DOWN':self.direction == Directions.DOWN}
        points = self.get_points()
        state = [
            #straight collision
            (direction_state['RIGHT'] and self.collides(points['RIGHT'])) or
            (direction_state['LEFT'] and self.collides(points['LEFT'])) or
            (direction_state['UP'] and self.collides(points['UP'])) or
            (direction_state['DOWN'] and self.collides(points['DOWN'])),
            #right collision
            (direction_state['RIGHT'] and self.collides(points['DOWN'])) or
            (direction_state['LEFT'] and self.collides(points['UP'])) or
            (direction_state['UP'] and self.collides(points['RIGHT'])) or
            (direction_state['DOWN'] and self.collides(points['LEFT'])),
            #left collision
            (direction_state['RIGHT'] and self.collides(points['UP'])) or
            (direction_state['LEFT'] and self.collides(points['DOWN'])) or
            (direction_state['UP'] and self.collides(points['LEFT'])) or
            (direction_state['DOWN'] and self.collides(points['RIGHT'])),

            #direction states
            direction_state['LEFT'],
            direction_state['RIGHT'],
            direction_state['UP'],
            direction_state['DOWN'],

            #food location
            self.food.x < self.head.x,
            self.food.x > self.head.x,
            self.food.y < self.head.y,
            self.food.y > self.head.y
        ]
        state = np.array(state, dtype=np.uint8)
        return state

    def play(self,action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        initial_distance = self.get_distance()
        self._move(action)
        final_distance = self.get_distance()
        alive=True    
        reward=0
        if final_distance < initial_distance:
            reward = 2
        if self.collides() or self.frame_iteration > 110*len(self.snake):
            alive=False
            reward = -50
        if self.head == self.food:
            reward = 10
            self.score+=1
            self._place_food()
            #self.frame_iteration=0
        else:
            self.snake.pop()
        
        self._update()
        self.clock.tick(const.SPEED)

        return reward,alive,self.score
    
    def collides(self,pt=None):
        if pt is None:
            pt=self.head
        if((pt.x > self.width - const.BLOCK_SIZE or pt.x < 0) or (pt.y > self.height - const.BLOCK_SIZE or pt.y < 0)):
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update(self):
        self.display.fill(const.COLORS['black'])
        for pt in self.snake:
            pygame.draw.rect(self.display,const.COLORS['blue'],pygame.Rect(pt.x,pt.y,const.BLOCK_SIZE,const.BLOCK_SIZE))
            pygame.draw.rect(self.display,const.COLORS['green'],pygame.Rect(pt.x+4,pt.y+4,12,12))
            #pygame.draw.aaline(self.display,const.COLORS['white'],(self.width/2,self.height/2),(pt.x,pt.y))
        pygame.draw.rect(self.display,const.COLORS['red'],pygame.Rect(self.food.x,self.food.y,const.BLOCK_SIZE,const.BLOCK_SIZE))
        text=const.FONT.render("Score : "+str(self.score),True,const.COLORS['white'])
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def _move(self,action):
        clock_wise=[Directions.RIGHT,Directions.DOWN,Directions.LEFT,Directions.UP]
        ind=clock_wise.index(self.direction)

        if np.array_equal(action,[1,0,0]):
            new_direction=clock_wise[ind]
        
        elif np.array_equal(action,[0,1,0]):
            new_ind=(ind+1)%4
            new_direction=clock_wise[new_ind]
        else:
            new_ind=(ind - 1)%4
            new_direction=clock_wise[new_ind]

        self.direction=new_direction  

        x,y=self.head.x,self.head.y
        if self.direction == Directions.RIGHT:
            x += const.BLOCK_SIZE
        elif self.direction == Directions.LEFT:
            x -= const.BLOCK_SIZE
        elif self.direction == Directions.UP:
            y += const.BLOCK_SIZE
        elif self.direction == Directions.DOWN:
            y -= const.BLOCK_SIZE
        
        self.head = const.Point(x,y)
        self.snake.insert(0,self.head)