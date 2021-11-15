import math

class Vector2D:
    # Vector2D class for Python 3
    # Original version from Learning Scientific Programming with Python, Second Edition, by Christian Hill, (ISBN: 9781108745918) in December 2020
    # This version overloads the operators +,-,* etc to allow them to work with
    # Vector2D operations.  See e.g. https://www.geeksforgeeks.org/operator-overloading-in-python/ for
    # an introduction to operator overloading in Python3.

    def __init__(self, x, y):
        # constructor method
        self.x, self.y = x, y

    def __str__(self):
        # Human-readable string representation of the vector.
        return '({:g},{:g})'.format(self.x, self.y)

    def dot(self, other):
        # The scalar (dot) product of self and other. Both must be vectors.
        if not isinstance(other, Vector2D):
            raise TypeError('Can only take dot product of two Vector2D objects')
        return self.x * other.x + self.y * other.y
        
    # Alias the __matmul__ method to dot so we can use a @ b as well as a.dot(b).
    __matmul__ = dot #  Overloads @ operator.

    def __sub__(self, other):
        # Vector subtraction.  Overloads - operator.
        return Vector2D(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        # Vector addition.  Overloads + operator.
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        # Multiplication of a vector by a scalar.  Overloads * operator.
        if isinstance(scalar, int) or isinstance(scalar, float):
            return Vector2D(self.x*scalar, self.y*scalar)
        raise NotImplementedError('Can only multiply Vector2D by a scalar')

    def __rmul__(self, scalar):
        # Reflected multiplication so vector * scalar also works.
        return self.__mul__(scalar)

    def __neg__(self):  
        # Negation of the vector.  Overloads - operator.
        return Vector2D(-self.x, -self.y)

    def __truediv__(self, scalar):
        # True division of the vector by a scalar.  Overloads / operator.
        return self*(1/scalar)

    def __abs__(self):
        # Absolute value (magnitude) of the vector.  Overloads abs function.
        return math.sqrt(self.x**2 + self.y**2)

    def mag(self): # magnitude of the vector
        return abs(self)

    def normalise(self): # return a normalised copy of this vector
        length=self.mag()
        if length!=0:
            return self/length
        else:
            return self

if __name__ == '__main__':
    v1 = Vector2D(2, 5/3)
    v2 = Vector2D(3, -1.5)
    print('v1 = ', v1)
    print('v1 + v2 = ', v1 + v2)
    print('v1 - v2 = ', v1 - v2)
    print('abs(v2 - v1) = ', abs(v2 - v1))
    print('-v2 = ', -v2)
    print('v1 * 3 = ', v1 * 3)
    print('7 * v2 = ', 7 * v1)
    print('v2 / 2.5 = ', v2 / 2.5)
    print('v1.dot(v2) = v1 @ v2 = ', v1 @ v2)
    print('v1.normalise()',v1.normalise())
