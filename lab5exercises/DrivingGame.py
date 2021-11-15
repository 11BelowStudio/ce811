# Neural-network Driving Game
# November 2021, Michael Fairbank, University of Essex, part of Module CE811
from typing import List, Tuple

import pygame, math, random, sys
from vector import Vector2D
from TrackLayout import TrackLayout
from enum import Enum     
import numpy as np


class SteeringAgent:

    def __init__(self, init_x, init_y, draw_colour, max_speed):
        self.agent_position = Vector2D(init_x,init_y)
        self.agent_velocity = Vector2D(0,0)
        self.agent_orientation = 0 # heading North initially
        self.max_speed=max_speed
        self.draw_colour=draw_colour
               
    def calculate_driving_decision(self, xte, road_angle,distance_around_track):
        # This function must return a pair of numbers, (target_speed, steering_wheel_position), with 0<=target_speed<=1 and -1<=steering_wheel_position<=1
        steering_wheel_position=0        
        target_speed=1
        return (target_speed, steering_wheel_position)
       
    def advance_agent(self,driving_speed, steering_wheel_position, deltaT):
        assert driving_speed<=1 and driving_speed>=0
        assert abs(steering_wheel_position)<=1
        # move the car forwards a little bit, according to the steering_wheel_position and driving speed.
        self.agent_orientation+=4*deltaT*steering_wheel_position
        self.agent_velocity=self.max_speed*driving_speed*Vector2D(math.sin(self.agent_orientation),math.cos(self.agent_orientation))
        self.agent_position += self.agent_velocity * deltaT    	
        
    def draw_agent_world_coords(self):
        size = (16, 16)
        temp_surface = pygame.Surface(size,pygame.SRCALPHA)
        colour=self.draw_colour+[255] # the 255 here is the alpha value, i.e. we want this polygon to be opaque
        pygame.draw.polygon(temp_surface, colour, ((14,16),(8,0),(2,16))) # draw a solid triangle shape pointing straight up
        ang=self.agent_orientation
        rotated_surface=pygame.transform.rotate(temp_surface, math.degrees(-ang)) # rotate anticlockwise by amount ang (need anticlockwise here because screencoords are flipped upside-down)
        agent_screen_coords=convert_to_screencoords(np.array([self.agent_position.x, self.agent_position.y]))[0]
        screen.blit(rotated_surface, (agent_screen_coords[0]-size[0]//2,agent_screen_coords[1]-size[1]//2))

        
class KeyboardSteeringAgent(SteeringAgent):
    def calculate_driving_decision(self,xte, road_angle, distance_around_track):
        # This function must return a pair of numbers, (target_speed, steering_wheel_position), with 0<=target_speed<=1 and -1<=steering_wheel_position<=1
        target_speed=1
        keys=pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            steering_wheel_position=-1
        elif keys[pygame.K_RIGHT]:
            steering_wheel_position=1
        else:
            steering_wheel_position=0
        return (target_speed, steering_wheel_position)

class AutonomousSteeringAgent(SteeringAgent):
    def calculate_driving_decision(self,xte, road_angle, distance_around_track):
        # This function must return a pair of numbers, (target_speed, steering_wheel_position),
        # with 0<=target_speed<=1 and -1<=steering_wheel_position<=1
        # Note that road_angle and self.agent_orientation are in radians.

        # By comparing them, and allowing for the fact that agent_orientation could have any multiple of 2pi added to it,
        # we can make the steering decision. However we must also consider the xte (cross-track error).
        # If abs(xte)>=track_layout.track_width then it means we've crashed off the road.
        # Look at xte and road_angle and self.agent_orientation+2n*pi to make a decision about which way to steer
        # If xte>0 then we should probably turn left  (depending on the orientation that the car is already in)
        # If xte<0 then we should probably turn right (depending on the orientation that the car is already in)

        current_orientation = self.agent_orientation % (2 * math.pi)
        steering_wheel_position = 0
        buffer_dist = track_layout.track_width/4
        target_speed = 1
        orientation_difference = current_orientation - road_angle

        if xte > buffer_dist:
            if orientation_difference < 0:
                steering_wheel_position = 1
            elif orientation_difference > 0:
                steering_wheel_position = -1

        elif xte < -buffer_dist:
            if orientation_difference < 0:
                steering_wheel_position = -1
            elif orientation_difference > 0:
                steering_wheel_position = 1


        return (target_speed, steering_wheel_position)

        
class NeuralSteeringAgent(SteeringAgent):
    ninputs=4
    nhids=6
    nouts=2
    weights_and_bias_shapes=[[ninputs,nhids],[nhids],[nhids,nouts],[nouts]]
    chromosome_length=ninputs*nhids+nhids+nhids*nouts+nouts
    
    def __init__(self, init_x, init_y, draw_colour, max_speed, chromosome):
        SteeringAgent.__init__(self, init_x, init_y, draw_colour, max_speed) 
        weights_and_biases=NeuralSteeringAgent.convert_vector_to_weights_and_biases(chromosome)
        [self.W1, self.b1, self.W2, self.b2]=weights_and_biases
        
       
    def calculate_driving_decision(self,xte, road_angle, distance_around_track):
        delta_angle = self.agent_orientation-road_angle
        delta_angle -= ((delta_angle+math.pi)//(2*math.pi))*(2*math.pi) # try and push the delta_angle to between -pi and +pi
        total_track_length=19785.189051733727
        track_width=900
        d=distance_around_track/total_track_length
        # Build an input vector containing information of what is probably important to making a driving decision.
        # The first two inputs give information about where we are on the track globally (could be useful in deciding which corners can be cut through by the car)
        # The next two inputs contain xte and driving angle compared to road - both rescaled to range [-1,1]

        input_vector=np.array([[math.sin(d),math.cos(d), xte/(track_width/2), np.tanh(delta_angle/(math.pi/16))]])

        # run a standard neural network with just one hidden layer, and tanh activation functions everywhere.
        # Note, we're doing this in Numpy, not tensorflow.  The functions matmul etc. have much the same names and do the same things.
        # But we don't get automatic-differentiation with Numpy, which we don't need anyway because we're using a GA to train our network.

        h1=np.tanh(np.matmul(input_vector,self.W1)+self.b1)
        output=np.tanh(np.matmul(h1,self.W2)+self.b2)

        # Hopefully the output will contain useful information to drive the car (it will if the neural network is trained properly!)

        steering_output=(output[0,1])
        speed_output=(output[0,0])/2+0.5 # rescale it from [-1,1] to [0,1]
        return speed_output, steering_output
        
    @staticmethod    
    def convert_vector_to_weights_and_biases(vec):
        # A useful helper function for converting a flat numpy vector into a list of the 4 weight and bias matrices required.
        # On entry, shapes should be a list e.g. [[4,5],[5],[5,2],[2]] or similar, specifying the shape of each weight/bais matrix in a neural network
        c=0
        weights_and_biases=[]
        for shape in NeuralSteeringAgent.weights_and_bias_shapes:
            if len(shape)==2:
                # we need to unpack this chunk into a matrix shape (it must be a weight matrix)
                [m,n]=shape
                s=m*n
                weights_and_biases.append(vec[c:c+s].reshape((m,n)))
                c+=s
            else:
                assert len(shape)==1
                # we need to unpack this chunk into a vector shape (it must be a bias vector)
                [m]=shape
                s=m
                weights_and_biases.append(vec[c:c+s])
                c+=s
        assert c==vec.shape[0] # check the vector given was exactly the correct length
        return weights_and_biases
        
            
def convert_to_screencoords(array):
    # helper function to convert measurements on physical track scale (i.e. mm) to screen pixels
    track_bottom_left_coord=track_layout.track_bounding_box[0]
    track_top_right_coord=track_layout.track_bounding_box[1]
    track_size=track_top_right_coord-track_bottom_left_coord
    array=array-track_bottom_left_coord
    # rescale all corrdinates to fit on screen
    array=array/track_size.max()/1.1*np.array([screen_size]).max()+np.array([[20,20]])
    # flip upside down so y=0 is at bottom of screen, not the top (hopefully more intuitive that way)
    array=array*np.array([[1,-1]])+np.array([[0,screen_size[1]]])
    return array.astype(np.int32)


def run_silently(track_layout, driving_agent, max_steps=1000, deltaT=1/50):
    # Run the car round the track without any graphics involved (useful for training with the G.A.)
    # The car runs until it either crashes into a wall, or max_steps is reached.  This function then returns the score (fitness) as total distance driven.
    cld = track_layout.calculateCarLocationDetails(driving_agent.agent_position.x, driving_agent.agent_position.y)
    initial_distance_around_track=cld.distance_around_track
    laps_completed=0
    previous_segment_number=cld.segment_number
    for step in range(max_steps):
        cld=track_layout.calculateCarLocationDetails2(driving_agent.agent_position.x, driving_agent.agent_position.y,previous_segment_number)
        if cld.segment_number<10 and previous_segment_number>track_layout.num_track_segments*0.9:
            laps_completed+=1
        elif cld.segment_number>track_layout.num_track_segments*0.9 and previous_segment_number<10:
            laps_completed-=1
        previous_segment_number=cld.segment_number
        driving_speed,steering_wheel_position=driving_agent.calculate_driving_decision(cld.xte, cld.road_angle, cld.distance_around_track)
        driving_agent.advance_agent(driving_speed,steering_wheel_position,deltaT)
        track_distance_driven=cld.distance_around_track-initial_distance_around_track+laps_completed*track_layout.total_track_length
        car_on_track=abs(cld.xte)<track_layout.track_width/2
        if not car_on_track:
            break
    return track_distance_driven


def genetic_algorithm_fitness(chromo: np.ndarray, step_count: int, track_layout: TrackLayout, deltaT: float) -> float:
    """
    Obtains the fitness of an individual we're testing with our genetic algorithm
    :param chromo: the chromosome for the individual
    :param step_count: timesteps for the test
    :param track_layout: track layout it's being tested on
    :param deltaT: timestep size for the test
    :return: fitness of that individual
    """
    return run_silently(
        track_layout,
        NeuralSteeringAgent(
            track_layout.preferred_start_point[0],
            track_layout.preferred_start_point[1],
            None,
            step_count,
            chromo
        ),
        step_count,
        deltaT
    )


def genetic_algorithm_runner(gens: int, restarts: int, chromo_length: int, mut: float, initial_range: float,
                             step_count: int,
                             track_layout: TrackLayout, deltaT: float, printouts: bool = True
                             ) -> List[Tuple[np.ndarray, float]]:
    """
    Runs the genetic algorithm stuff
    :param gens: how many generations are we running?
    :param restarts: how many restarts will we be doing?
    :param chromo_length: how long is the chromosome?
    :param mut: what's the mutation rate?
    :param initial_range: range of the values that the first individual will be initialized with
    :param step_count: timestep count to be used when evaluating fitness
    :param track_layout: track layout we're testing it on
    :param deltaT: deltaT for the test
    :param printouts: set this to true if we want printouts from this test
    :return: A list of the best chromosomes produced by the genetic algorithm sorted by fitness
    """
    fittests: List[Tuple[np.ndarray, float]] = []
    for r in range(restarts):
        current_individual: np.ndarray = np.clip(
            np.random.normal(loc=0, scale=initial_range, size=(chromo_length)),
            -1.0,
            1.0
        )

        fitness: float = genetic_algorithm_fitness(current_individual, step_count, track_layout, deltaT)

        for g in range(gens):
            next_individual: np.ndarray = np.clip(
                current_individual + np.random.normal(loc=0, scale=mut, size=(chromo_length)),
                -1.0,
                1.0
            )
            next_fitness: float = genetic_algorithm_fitness(next_individual, step_count, track_layout, deltaT)

            if next_fitness > fitness:
                current_individual = next_individual
                fitness = next_fitness

        fittests.append((current_individual, fitness))
        if printouts:
            print("Round {} best fitness: {}".format(r, fitness))

    fittests.sort(key=lambda kv: kv[1], reverse=True)
    if printouts:
        print("")
        print("Fittest individuals:")
        print("Fitness: chromosome")
        for f in fittests:
            print("{}: {}".format(np.format_float_scientific(f[1], precision=3, exp_digits=3), f[0].tolist()))
    return fittests




if __name__=="__main__":
    deltaT=1/50
    track_layout=TrackLayout()
    from enum import Enum

    class Agents(Enum):
        KEYBOARD = 1
        AUTONOMOUS = 2
        NEURAL = 3

    agent=Agents.NEURAL # TODO modify this line to switch agent
    
    
    if agent==Agents.NEURAL:
        # For simplicity, choose the "stochastic hill climber"
        # See lecture slides on "stochastic hill climber"
        # Use the "run_silently" function to evaluate the fitness of a chromosome.  This runs the car around the track quickly and returns the distance travelled as the fitness.
        # If necessary then see also https://machinelearningmastery.com/stochastic-hill-climbing-in-python-from-scratch/
        # For the mutation operator, I found that just adding an array of randomly distributed noise (normally distributed with standard deviation 0.05) worked quite well.  See help on np.random.normal for ideas.
        # You might need more trials than 200, and you might need several random restarts (just re-run the program yourself to do this).

        # hyper-parameters for this GA run
        generations: int = 400
        attempts: int = 10
        mutation_deviation: float = 0.05
        initial_deviation: float = 2
        test_timesteps: int = 1000
        printouts: bool = True

        # And here we actually run the GA stuff.
        fittests: List[Tuple[np.ndarray, float]] = genetic_algorithm_runner(
            generations,
            attempts,
            NeuralSteeringAgent.chromosome_length,
            mutation_deviation, initial_deviation,
            test_timesteps,
            track_layout,
            deltaT,
            printouts
        )

        # after your GA has run, we should have built a good chromosome here....
        print("Best chromosome: {}: {}".format(fittests[0][1], fittests[0][0].tolist()))

        chromosome: np.ndarray = fittests[0][0]
        
    pygame.init()
    pygame.font.init()
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    screen_size = [500,600]
    black=[0,0,0]
    screen=pygame.display.set_mode(screen_size)
    pygame.display.set_caption("CE811 Neuro-driver")
    clock = pygame.time.Clock()

    # build graphical representation of the track layout...
    yellow = [255,255,0]
    white=[255,255,255]
    track_inside_lane_polyline=convert_to_screencoords(np.stack([track_layout.getCoordinatesRelativeToSegmentPosition(segNum, 0, -track_layout.track_width/2) for segNum in range(track_layout.num_track_segments)]))
    track_outside_lane_polyline=convert_to_screencoords(np.stack([track_layout.getCoordinatesRelativeToSegmentPosition(segNum, 0, +track_layout.track_width/2) for segNum in range(track_layout.num_track_segments)]))
    track_centre_lane_polyline=convert_to_screencoords(np.stack([track_layout.getCoordinatesRelativeToSegmentPosition(segNum, 0, +0) for segNum in range(track_layout.num_track_segments)]))
    
    # repeatedly apply the trained agent in an animation...
    while True:
        if agent==Agents.KEYBOARD:
            driving_agent = KeyboardSteeringAgent(track_layout.preferred_start_point[0],track_layout.preferred_start_point[1], yellow,1000)
        elif agent==Agents.AUTONOMOUS:
            driving_agent = AutonomousSteeringAgent(track_layout.preferred_start_point[0],track_layout.preferred_start_point[1], yellow,1000)  
        elif agent==Agents.NEURAL:
            driving_agent = NeuralSteeringAgent(track_layout.preferred_start_point[0],track_layout.preferred_start_point[1], yellow,1000,chromosome)


        cld=track_layout.calculateCarLocationDetails(driving_agent.agent_position.x, driving_agent.agent_position.y)
        initial_distance_around_track=cld.distance_around_track
        laps_completed=0
        previous_segment_number=cld.segment_number

        done=False
        while done==False:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done=True
                    pygame.quit()
                    sys.exit(0)

            screen.fill(black) # background screen colour
            # Draw track sides:    
            pygame.draw.polygon(screen, white, track_inside_lane_polyline, 5)
            pygame.draw.polygon(screen, white, track_outside_lane_polyline, 5)
            pygame.draw.polygon(screen, white, track_centre_lane_polyline, 1)
            
            # find out where car is on track            
            cld=track_layout.calculateCarLocationDetails2(driving_agent.agent_position.x, driving_agent.agent_position.y,previous_segment_number)
            if cld.segment_number<10 and previous_segment_number>track_layout.num_track_segments*0.9:
                laps_completed+=1
            elif cld.segment_number>track_layout.num_track_segments*0.9 and previous_segment_number<10:
                laps_completed-=1
            previous_segment_number=cld.segment_number
            
            # apply driving decision
            driving_speed,steering_wheel_position=driving_agent.calculate_driving_decision(cld.xte, cld.road_angle, cld.distance_around_track)
            driving_agent.advance_agent(driving_speed,steering_wheel_position,deltaT)
            driving_agent.draw_agent_world_coords()

            track_distance_driven=cld.distance_around_track-initial_distance_around_track+laps_completed*track_layout.total_track_length
            #print("cld",car_on_track,cld.segment_number,laps_completed, track_distance_driven)
            textsurface = myfont.render('Score: '+str(round(track_distance_driven))+" xte:"+str(round(cld.xte))+" steer:"+str(round(steering_wheel_position,1))+" accelerator:"+str(round(driving_speed,1)), False, white)
            screen.blit(textsurface,(0,0))
            pygame.display.flip() # pushes all drawings to the screen
            clock.tick(1/deltaT) # pauses deltaT time
            
            car_on_track=abs(cld.xte)<track_layout.track_width/2
            if not car_on_track:
                done=True
        print("Final score",track_distance_driven)
    pygame.quit()
