import os
from glob import glob

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "dataset/", "dataset dir path")


def main(argv):
    dir_list = glob(os.path.join(FLAGS.data_dir, "*"))
    for d in dir_list:
        if d.endswith("B"):
            file_list = glob(os.path.join(d, "*"))
            for f in file_list:
                os.rename(f, f[:-3] + "png")
    print("Done!")


if __name__ == "__main__":
    app.run(main)