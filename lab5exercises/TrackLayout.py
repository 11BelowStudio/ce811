import numpy as np
import math

# Initial C version by Ian Dukes, University of Essex 2015
# Converted to python for CE811 M. Fairbank 2021

# These are the physical measurements from the track layout in the Vicon arena at Essex university. All measurements in mm.
centreLine1=np.array([-1510,-1429, -1405,-1075, -1267,-723, -1065,-389, -846,-67, -656,271, -487,609, -394,948, 
        -382,1285, -414,1570, -534,1781, -724,1976, -939,2192, -1150,2434, -1340,2694, -1415,2979, -1391,3280, -1286,3578, 
        -1119,3823, -888,3999, -636,4104, -362,4170, -110,4170, 126,4129, 394,4035, 648,3897, 896,3694, 1132,3459, 1329,3203, 
        1489,2909, 1616,2630, 1734,2333, 1791,2035, 1827,1717, 1854,1373, 1849,1031, 1830,684, 1819,337, 
        1800,-276, 1772,-1379, 1739,-1791, 1706,-2182, 1672,-2570, 1557,-2915, 1363,-3177, 
        1118,-3363, 792,-3464, 417,-3502, -1,-3500, -413,-3436, -778,-3295, -1071,-3083, -1289,-2795, -1424,-2473, 
        -1533,-2124, -1567,-1778],np.float32).reshape([-1,2])*np.array([[1,-1]])


class TrackLayout:
 
    def __init__(self, ta=centreLine1[:,:]):
        self.trackCentreCoordArray=ta
        lengthened_track_array=np.concatenate([self.trackCentreCoordArray,self.trackCentreCoordArray[0:1,:]],axis=0)
        numberOfCoords = ta.shape[0]
        segment_displacements=lengthened_track_array[1:,:]-lengthened_track_array[:-1,:]
        self.segment_lengths=np.sqrt(np.square(segment_displacements).sum(axis=1)) 
        self.distance_around_track_to_segment_ends=np.cumsum(self.segment_lengths,axis=0) # calculates cumulative sum of the whole array
        self.distance_around_track_to_segment_starts=np.concatenate([np.array([0.0]),self.distance_around_track_to_segment_ends],axis=0)
        self.total_track_length=self.distance_around_track_to_segment_ends[-1]
        self.segment_unit_tangent_vectors=segment_displacements/np.expand_dims(self.segment_lengths,1)
        self.segment_unit_normal_vectors=-np.stack([-self.segment_unit_tangent_vectors[:,1],self.segment_unit_tangent_vectors[:,0]],axis=1) # rotates each vector 90 degrees
        self.num_track_segments=self.trackCentreCoordArray.shape[0]
        self.track_width=900# all units in mm
        self.track_bounding_box=self.getTrackBoundingBox()
        self.preferred_start_point=np.array([1700,1500])
        self.segment_orientations=np.arctan2(self.segment_unit_tangent_vectors[:,0],self.segment_unit_tangent_vectors[:,1])
        self.preferredStartTheta=-175
    
    def getCoordinatesRelativeToSegmentPosition(self,segment_number, sub_length, xte):
        segment_length=self.segment_lengths[segment_number]
        assert sub_length<segment_length+1e-2
        #assert sub_length>-1e-1 # TODO restore
        assert segment_length>1e-4
        segment_start_position = self.trackCentreCoordArray[segment_number,:]
        segment_direction=self.segment_unit_tangent_vectors[segment_number,:]
        segment_normal=self.segment_unit_normal_vectors[segment_number,:]
        assert abs(np.dot(segment_direction,segment_direction)-1)<1e-6
        assert abs(np.dot(segment_normal,segment_normal)-1)<1e-6
        # calc displacement vector of bx,by from the end x1,y1
        pos=segment_start_position+segment_direction*sub_length+xte*segment_normal
        
        #cld=self.carLocationRelativeToSegment(pos, segment_number)
        #print("cld.segment_number",cld.segment_number, segment_number)
        #print("cld.xte",cld.xte, xte)
        #print("cld.sub_length",cld.sub_length, sub_length)
        #print()
        return pos
    
    
    def getCoordinatesRelativeToCumulativeTrackLengthPosition(self,distance_around_track, xte):
        assert distance_around_track>=0 
        assert distance_around_track<=self.total_track_length
        if distance_around_track>=self.total_track_length-1e-2:
            distance_around_track=self.total_track_length-1e-2
        index=np.argmax(self.distance_around_track_to_segment_ends>distance_around_track) # find index of track-segment in which distance_around_track lies (https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value)
        assert self.distance_around_track_to_segment_ends[index]>distance_around_track
        if index>0:
            assert self.distance_around_track_to_segment_ends[index-1]<distance_around_track
            segment_distance=self.distance_around_track_to_segment_ends[index-1]
        else:
            segment_distance=0
        sub_length=distance_around_track-segment_distance
        if not sub_length<=self.segment_lengths[index] or not sub_length>=0:
            print("index found",index)
            print("distance_around_track",distance_around_track,index,sub_length,self.segment_lengths[index])
            print("self.total_track_length",self.total_track_length)
            
            print("self.distance_around_track_to_segment_ends",self.distance_around_track_to_segment_ends)
        assert sub_length<=self.segment_lengths[index]
        assert sub_length>=0
        return self.getCoordinatesRelativeToSegmentPosition(index, sub_length, xte)
        

    def getTrackBoundingBox(self):
        minXY=self.trackCentreCoordArray[:,:].min(axis=0)
        maxXY=self.trackCentreCoordArray[:,:].max(axis=0)
        return [minXY-self.track_width/2,maxXY+self.track_width/2]
    

    #find which sector the car is in. This is necessary as when the code starts it has to loacate the car relative
    # to the track sector. 
    #see pic
    # car is located by calculated shortest distance to a wp node, the looking at the previous and following
    #wp nodes it can be determined if a wp node has been passed and hence locating which sector around the 
    #track the car is currently in. 

    def calculateCarLocationDetails(self,carX, carY):
        #find shortest distance from car to a wp node
        approximate_track_segment_index=self.identify_closest_track_node(np.array([carX,carY]))
        return self.calculateCarLocationDetails2(carX, carY, approximate_track_segment_index)
 
    def calculateCarLocationDetails2(self,carX, carY, old_segment_number):
        num_track_segments=self.trackCentreCoordArray.shape[0]
        car_location=np.array([carX,carY])
        index=(old_segment_number+num_track_segments-1)%num_track_segments
        for i in range(3):
            cld = self.carLocationRelativeToSegment(car_location, index)
            if i==0:
                assert cld.sub_length>1e-2
            if (cld.sub_length<self.segment_lengths[cld.segment_number]):
                '''if not(cld.sub_length>-10):
                    print("car",carX,carY)
                    print("i",i, "cld.sub_length",cld.sub_length,"xte",cld.xte)
                    print("old_segment_number",old_segment_number,"index",index)
                assert cld.sub_length>-10 #TODO put this back and figure it out'''
                return cld
            index=(index+1)%num_track_segments
        raise Exception("carX="+carX+" carY="+carY)
 
    
    def identify_closest_track_node(self,car_location):
        dist_squared_car_to_track=np.square(self.trackCentreCoordArray-np.expand_dims(car_location,0)).sum(axis=1)
        closest_segment_number=np.argmin(dist_squared_car_to_track)
        return closest_segment_number
            
    def carLocationRelativeToSegment(self,car_location, segment_number):
        assert len(car_location.shape)==1
        assert car_location.shape[0]==2
        num_track_segments = self.trackCentreCoordArray.shape[0]
        dirxy=self.segment_unit_tangent_vectors[segment_number]
        dirxy_normal=self.segment_unit_normal_vectors[segment_number]
        # calc displacement vector from start of track segment to car.
        car_displacement_along_segment=car_location-self.trackCentreCoordArray[segment_number]
        # project this displacement vector onto the line...
        projection=np.dot(dirxy,car_displacement_along_segment)
        xte=np.dot(car_displacement_along_segment,dirxy_normal)
        cumulativeLength=self.distance_around_track_to_segment_starts[segment_number]
        return CarLocationDetailsWRTtrack(projection, xte, cumulativeLength+projection, segment_number,self.segment_orientations[segment_number])            
        
    def isPointOnTrack(self, carX,carY):
        cld = self.calculateCarLocationDetails(carX, carY)
        return abs(cld.xte)<=self.track_width/2;
    
    
class CarLocationDetailsWRTtrack:
    # structure to hold useful information about where the car is on the track
    def __init__(self, sub_length, xte, distance_around_track, segment_number, road_angle):
        self.sub_length = sub_length # car's position projected along segment's centre line
        self.xte = xte # cross-track error, i.e. car's position projected perpendicular to centre line of segment.
        self.distance_around_track = distance_around_track # total distance that car is around the track.
        self.segment_number = segment_number
        self.road_angle = road_angle
        
