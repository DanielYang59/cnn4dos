"""Load eDOS dataset (feature/label), based on config file."""

# NOTE:
# Need to be thoughtful not to directly load the file to memory,
# but use TF to load them on the fly to reduce memory footage.


class Dataset:
    def load_label(self):
        pass

    def load_feature(self):
        pass
