def constant1d(t, x0, speed):
    """
    Calculate the position of an object moving at constant speed at time t.
    :param speed: Speed of the moving object
    :param t: time
    :param x0: position of the object at time 0
    :return: x(t)
    """
    return x0 + speed * t
