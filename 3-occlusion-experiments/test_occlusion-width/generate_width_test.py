#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import yaml
import shutil


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    occlusion_script = cfg["path"]["occlusion_script"]
    example_dos = cfg["path"]["example_dos"]
    widths = cfg["test"]["widths"]


    for width in widths:
        # Create result dir
        result_dir = os.path.join("results", f"width_{width}")
        os.makedirs(result_dir, exist_ok=True)


        # Copy source DOS array
        shutil.copyfile(example_dos, os.path.join(result_dir, "dos_up.npy"))


        # Load occlusion generation script
        with open(os.path.join(occlusion_script)) as f:
            script = f.readlines()

        # Modify width in occlusion generation script
        new_script = []
        for line in script:
            if line.replace(" ", "").startswith("masker_width="):
                line = f"masker_width = {width}\n"
            new_script.append(line)

        # Write new script
        with open(os.path.join(result_dir, occlusion_script.split(os.sep)[-1]), mode="w") as f:
            f.writelines(new_script)


    print("All occlusion width tests generated.")
