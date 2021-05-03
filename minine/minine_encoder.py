import numpy as np
import torch
import yaml
from scipy.spatial import ConvexHull

from minine.codec import Encoder
from minine.frame_sequence import EncodedFrame
from minine.latent_representation import encID, encPE
from minine.modules.keypoint_detector import KPDetector
from minine.resources import get_config_path, get_weights_path

# from sync_batchnorm import DataParallelWithCallback #TODO this now assumes 0 or 1 GPU

# TODO write a new custom loader
config_path = get_config_path()
checkpoint_path = get_weights_path()


class MinineEncoder(Encoder):
    def __init__(self, identity_refresh_rate=50):

        self.identity_refresh_rate = identity_refresh_rate

        # setting device on GPU if available, else CPU
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Internal state
        self.need_initial_frame = True
        self.identity_refresh_count = 0
        self.last_identity_frame = None
        self.last_identity_frame_kp = None

        self.kp_detector = self.load_kp_detector()

    def encode_frame(self, frame):
        """
        This method takes one frame, encodes it and adds it to
        the output buffer
        """
        with torch.no_grad():

            if (
                self.need_initial_frame
                or self.identity_refresh_count >= self.identity_refresh_rate
            ):
                self.need_initial_frame = False

                # load the source frame as a tensor
                identity_frame = torch.tensor(np.array(frame).astype(np.float32))
                identity_frame = (
                    identity_frame.unsqueeze(0).permute(0, 3, 1, 2).to(self.dev)
                )

                # TODO right now we also store the keypoints from the source frame, we don't need to do this
                self.last_identity_frame = identity_frame
                self.last_identity_frame_kp = self.kp_detector(
                    self.last_identity_frame / 255
                )  # TODO better normailsation

                self.identity_refresh_count = 0

                return EncodedFrame(
                    encID=encID(
                        self.last_identity_frame.cpu(), self.last_identity_frame_kp
                    )
                )

            else:
                driving_frame = torch.tensor(
                    np.array(frame).astype(np.float32) / 255
                )  # TODO better normalising
                driving_frame = driving_frame.unsqueeze(0).permute(0, 3, 1, 2)
                driving_frame = driving_frame.to(self.dev)
                driving_frame_kp = self.kp_detector(driving_frame)

                kp_norm = self.normalize_kp(
                    kp_source=self.last_identity_frame_kp,
                    kp_driving=driving_frame_kp,
                    kp_driving_initial=self.last_identity_frame_kp,
                    use_relative_movement=False,  # TODO We'll always use relative
                    use_relative_jacobian=False,  # TODO We probably don't need this
                    adapt_movement_scale=False,  # TODO We probably don't need this
                )

                self.identity_refresh_count += 1

                return EncodedFrame(encPE=encPE(kp_norm))

    def encode_sequence(self, sequence):
        # TODO use batch of frames instead of one (so don't use encode_frame())
        # TODO use GOP instead of lists
        encoded_sequence = []

        for frame in sequence:
            encoded_frame = self.encode_frame(frame)
            encoded_sequence.append(encoded_frame)

        return encoded_sequence

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

        kp_detector.to(self.dev)

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
            kp_detector.load_state_dict(checkpoint["kp_detector"])
            # kp_detector = DataParallelWithCallback(kp_detector) #TODO this now assumes 0 or 1 GPU
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            kp_detector.load_state_dict(checkpoint["kp_detector"])

        kp_detector.eval()

        return kp_detector

    def normalize_kp(
        self,
        kp_source,
        kp_driving,
        kp_driving_initial,
        adapt_movement_scale,
        use_relative_movement,
        use_relative_jacobian,
    ):
        """
        # normalise the keypoints of the driving vector
        """

        if adapt_movement_scale:
            source_area = ConvexHull(kp_source["value"][0].data.cpu().numpy()).volume
            driving_area = ConvexHull(
                kp_driving_initial["value"][0].data.cpu().numpy()
            ).volume
            adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
        else:
            adapt_movement_scale = 1

        kp_new = {k: v for k, v in kp_driving.items()}

        if use_relative_movement:
            kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
            kp_value_diff *= adapt_movement_scale
            kp_new["value"] = kp_value_diff + kp_source["value"]

            if use_relative_jacobian:
                jacobian_diff = torch.matmul(
                    kp_driving["jacobian"],
                    torch.inverse(kp_driving_initial["jacobian"]),
                )
                kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

        return kp_new
