class encID:
    """
    TODO might be usefull to make this an abstract class
    TODO we don't need to store the keypoints
    """

    def __init__(self, identity_frame, keypoints):
        self.keypoints = keypoints
        self.identity_frame = identity_frame


class encPE:
    """
    TODO might be usefull to make this an abstract class
    """

    def __init__(self, keypoints):
        self.keypoints = keypoints
