import numpy as np
import torch
import yaml

from minine.codec import Decoder
from minine.modules.generator import OcclusionAwareGenerator
from minine.modules.keypoint_detector import KPDetector
from minine.resources import get_config_path, get_weights_path

# TODO write a new custom loader
config_path = get_config_path()
checkpoint_path = get_weights_path()


class MinineDecoder(Decoder):
    def __init__(self):
        # setting device on GPU if available, else CPU
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO right now we also store the keypoints from the source frame, we don't need to do this
        #           i.e. we don't use the kp_detector atm
        # self.kp_detector = self.load_kp_detector()
        self.generator = self.load_generator()

        self.last_identity_frame = None
        self.last_identity_frame_kp = None

    def decode_frame(self, Frame):
        """
        This function decodes a single frame and manages the state of the decoder
        # TODO check whether the input matches the state of the decoder
        """

        # The decoder receives an I-frame, so we refresh the identity
        if Frame.encID != None:
            # load the source frame as a tensor
            identity_frame = torch.tensor(
                np.array(Frame.encID.identity_frame).astype(np.float32)
            )
            identity_frame = identity_frame.to(self.dev)

            # TODO right now we also store the keypoints from the source frame, we don't need to do this
            self.last_identity_frame = identity_frame / 255  # TODO better normalization
            self.last_identity_frame_kp = Frame.encID.keypoints
            # self.last_identity_frame_kp = self.kp_detector(self.last_identity_frame)

            identity_frame = np.uint8(identity_frame.squeeze(0).cpu().permute(1, 2, 0))

            return identity_frame

        # Identity is up-to-date, start decoding the motion vector
        decoded_frame = self.generator(
            self.last_identity_frame,
            kp_source=self.last_identity_frame_kp,
            kp_driving=Frame.encPE.keypoints,
        )

        decoded_frame = decoded_frame["prediction"].data
        decoded_frame = np.uint8(
            (256 * decoded_frame).squeeze(0).cpu().permute(1, 2, 0)
        )  # TODO better unnormailzation

        return decoded_frame

    def decode_sequence(self, sequence):
        # TODO use batch of frames instead of one (so don't use decode_frame())
        # TODO use GOP instead of lists
        decoded_sequence = []

        for frame in sequence:
            decoded_frame = self.decode_frame(frame)
            decoded_sequence.append(decoded_frame)

        return decoded_sequence

    def load_generator(self):
        """
        Initialising keypoint detector module from FOMMFIA
        """

        # TODO write a new custom loader
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        generator = OcclusionAwareGenerator(
            **config["model_params"]["generator_params"],
            **config["model_params"]["common_params"]
        )

        generator = generator.to(self.dev)

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
            generator.load_state_dict(checkpoint["generator"])
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            generator.load_state_dict(checkpoint["generator"])

        generator.eval()

        return generator

    def load_kp_detector(self):
        """
        Initialising keypoint detector module from FOMMFIA
        """

        # TODO write a new custom loader
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        kp_detector = KPDetector(
            **config["model_params"]["kp_detector_params"],
            **config["model_params"]["common_params"]
        )

        kp_detector = kp_detector.to(self.dev)

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
            kp_detector.load_state_dict(checkpoint["kp_detector"])
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            kp_detector.load_state_dict(checkpoint["kp_detector"])

        kp_detector.eval()

        return kp_detector
