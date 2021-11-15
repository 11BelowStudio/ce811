import pygame, math, random
from vector import Vector2D

from typing import List

pygame.init()
screen_size = [900,600]
black=[0,0,0]
screen=pygame.display.set_mode(screen_size)
pygame.display.set_caption("Steering Behaviours Demo")
pygame.mouse.set_visible(0)
done = False
mouse_down = False
clock = pygame.time.Clock()

class SteeringAgent:

    def __init__(self, init_x: float, init_y: float, draw_colour: List[int], max_speed: float, max_acc: float, mass: float):
        self.agent_position: Vector2D = Vector2D(init_x,init_y)
        self.agent_velocity: Vector2D = Vector2D(0,0)
        self.agent_mass: float = mass
        self.max_speed: float = max_speed
        self.max_acc: float = max_acc
        self.draw_colour: List[int] = draw_colour

    def apply_steering_force(self, desired_velocity: Vector2D, deltaT: float) -> None:
        steering_force = desired_velocity-self.agent_velocity
        steering_acceleration = steering_force / self.agent_mass
        if abs(steering_acceleration) > self.max_acc:
            steering_acceleration = steering_acceleration.normalise()*self.max_acc
        self.agent_position += self.agent_velocity * deltaT
        self.agent_velocity += steering_acceleration * deltaT
    
    def calculate_seek_velocity(self, target_position: Vector2D) -> Vector2D:
        # calculate the desired_velocity so as to "seek" to target position
        desired_velocity: Vector2D = target_position - self.agent_position
        return desired_velocity.normalise() * self.max_speed

    def calculate_flee_velocity(self, target_position: Vector2D) -> Vector2D:
        # calculate the desired_velocity so as to "flee" from the target position
        desired_velocity: Vector2D = self.agent_position - target_position
        return desired_velocity.normalise() * self.max_speed
       
    def calculate_pursuit_advanced_target(self, other_agent: "SteeringAgent", maxT: float) -> Vector2D:
        # calculate the position of the point in front of other_agent, according to the "pursuit" behaviour
        t: float = min(((other_agent.agent_position - self.agent_position)/self.max_speed).mag(), maxT)
        other_vel: Vector2D = other_agent.agent_velocity
        return other_agent.agent_position + (other_vel * t)

    def adjust_velocity_for_arrival(self, desired_velocity: Vector2D, target_position: Vector2D,
                                    arrivalSlowingDist: float, arrivalStoppingDist: float) -> Vector2D:
        # calculate a new velocity vector, based upon desired_velocity, but which is modified according to the "arrival" behaviour
        # On entry, desired_velocity will have magnitude self.max_speed
        # Calculate distance to the target.
        dist_left: float = (target_position - self.agent_position).mag()

        if dist_left < arrivalSlowingDist:
            if dist_left <= arrivalStoppingDist:
                # stop within the arrival stopping distance
                return Vector2D(0, 0)
            slowing_range: float = arrivalSlowingDist - arrivalStoppingDist
            dist_left -= arrivalStoppingDist
            return desired_velocity * (dist_left/slowing_range)
        return desired_velocity
       
    def get_agent_orientation(self) -> float:
        if self.agent_velocity.mag() > 0:
            ang = math.atan2(self.agent_velocity.x, self.agent_velocity.y)
            # measure angle of agent's velocity vector, in radians clockwise from north
        else:
            ang = 0
        return ang

    @property
    def orientation(self) -> float:
        """version of get_agent_orientation as a property instead so it's less verbose"""
        return self.get_agent_orientation()
        
    def keep_within_screen_bounds(self) -> None:
        self.agent_position.x = (self.agent_position.x+screen_size[0])%screen_size[0] # force wrap-around in x direction
        self.agent_position.y = (self.agent_position.y+screen_size[1])%screen_size[1] # force wrap-around in x direction
            
    def draw_agent(self) -> None:
        size = (16, 16)
        temp_surface = pygame.Surface(size, pygame.SRCALPHA)
        colour = self.draw_colour+[255]  # the 255 here is the alpha value, i.e. we want this polygon to be opaque
        pygame.draw.polygon(temp_surface, colour, ((14, 0), (8, 16), (2, 0)))
        # draw a solid triangle shape pointing straight up
        ang = self.get_agent_orientation()
        rotated_surface = pygame.transform.rotate(temp_surface, math.degrees(ang))  # rotate anticlockwise by amount ang
        screen.blit(rotated_surface, (self.agent_position.x, self.agent_position.y))


class WanderingAgent(SteeringAgent):  # inherits most behaviour from SteeringAgent

    def __init__(self, init_x: float, init_y: float, draw_colour: List[int], max_speed: float, max_acc: float,
                 mass: float, wander_rate: float, wander_offset: float, wander_radius: float):
        self.wander_rate: float = wander_rate
        self.wander_radius: float = wander_radius
        self.wander_offset: float = wander_offset
        self.wander_orientation: float = 0
        SteeringAgent.__init__(self, init_x, init_y, draw_colour, max_speed, max_acc, mass)

    def calculate_wander_seek_target(self, random_float: float) -> Vector2D:
        # This function needs to apply the main logic of the "wander" behaviour 
        # It should calculate the position that will then be used as the seek_target.
        # This function should return a position (of the seek target)
        # It should also modify self.wander_orientation
        # On entry: random_float is a random float in range from -1 to +1

        self.wander_orientation += random_float * self.wander_rate

        wander_target_angle: float = self.wander_orientation + self.orientation

        wander_target_circle: Vector2D = self.agent_position + (self.agent_velocity.normalise() * self.wander_offset)
        #(self.agent_velocity * self.wander_offset)
        #  self.make_polar_vector(self.orientation, self.wander_offset)

        seek_target: Vector2D = wander_target_circle + self.make_polar_vector(wander_target_angle, self.wander_radius)
        return seek_target

    def calc_wander_target_velocity(self) -> Vector2D:
        seek_target: Vector2D = self.calculate_wander_seek_target(random.uniform(-1, 1))
        # Seek towards the target small circle:
        return self.calculate_seek_velocity(seek_target)

    @property
    def orientation(self) -> float:
        """version of get_agent_orientation as a property instead so it's less verbose"""
        return self.get_agent_orientation()

    def make_polar_vector(self, angle: float, mag: float) -> Vector2D:
        """Creates a cartesian vector (x, y) from a polar vector (vector with angle and magnitude)"""
        return Vector2D(mag * math.sin(angle), mag * math.cos(angle))
    

def draw_mouse_pointer(screen,mouse_pos, colour):
    pygame.draw.rect(screen, colour, pygame.Rect(mouse_pos.x, mouse_pos.y, 12, 12))


yellow = [255, 255, 0]
magenta = [255, 0, 255]
blue = [0, 0, 255]
red = [255, 0, 0]
white = [255, 255, 255]

deltaT = 1/50
agent_seek = SteeringAgent(200, 200, magenta, 200, 300, .1)
agent_flee = SteeringAgent(300, 200, blue, 200, 300, .1)
agent_pursuit = SteeringAgent(300, 200, yellow, 200*2, 300*2, .1)
agent_wander = WanderingAgent(300, 200, white, 100, 300, .1, 20*deltaT, 20, 3)
mouse_pos = Vector2D(100,100)

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill(black)  # background screen colour
    
    mouse_down = pygame.mouse.get_pressed()[0]  # note: returns 0/1, which == False/True
    if not mouse_down:
        pos = pygame.mouse.get_pos()
        mouse_pos = Vector2D(pos[0],pos[1])
    draw_mouse_pointer(screen, mouse_pos, red)
    target_position = mouse_pos

    desired_velocity = agent_seek.calculate_seek_velocity(target_position)
    desired_velocity = agent_seek.adjust_velocity_for_arrival(desired_velocity, target_position, arrivalSlowingDist=40, arrivalStoppingDist=10)
    agent_seek.apply_steering_force(desired_velocity, deltaT)
    
    desired_velocity = agent_flee.calculate_flee_velocity(target_position)
    agent_flee.apply_steering_force(desired_velocity, deltaT)
    
    advanced_target=agent_pursuit.calculate_pursuit_advanced_target(agent_seek, maxT=6)
    desired_velocity=agent_pursuit.calculate_seek_velocity(advanced_target)
    agent_pursuit.apply_steering_force(desired_velocity, deltaT)
    
    desired_velocity=agent_wander.calc_wander_target_velocity()
    agent_wander.apply_steering_force(desired_velocity, deltaT)

    for agent in [agent_seek, agent_flee, agent_pursuit, agent_wander]:
        agent.keep_within_screen_bounds()
        agent.draw_agent()
        
    pygame.display.flip()  # pushes all drawings to the screen
    clock.tick(1/deltaT)  # pauses deltaT time

pygame.quit()
