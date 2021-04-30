import pathlib


def get_weights_path():
    return str(
        pathlib.Path(__file__).parent / "modules" / "weights" / "vox-cpk.pth.tar"
    )


def get_config_path():
    return str(pathlib.Path(__file__).parent / "modules" / "config" / "vox-256.yaml")
