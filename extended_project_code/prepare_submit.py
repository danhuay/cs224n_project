# Creates a zip file for submission on Gradescope.

import os
import zipfile
import click


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["dev", "test", "full"]),
    default="dev",
    help="Which files to zip",
)
def main(mode):
    py_files = [p for p in os.listdir(".") if p.endswith(".py")] + [
        f"src/{p}" for p in os.listdir("src") if p.endswith(".py")
    ] + [p for p in os.listdir(".") if p.endswith(".yml")]
    dev_files = [f"predictions/{p}" for p in os.listdir("predictions") if "dev" in p]
    test_files = [f"predictions/{p}" for p in os.listdir("predictions") if "test" in p]

    if mode == "dev":
        required_files = dev_files
    elif mode == "test":
        required_files = test_files
    elif mode == "full":
        required_files = py_files + dev_files + test_files
    else:
        raise ValueError("Invalid mode")

    aid = f"cs224n_default_final_project_submission_{mode}"

    with zipfile.ZipFile(f"{aid}.zip", "w") as zz:
        for file in required_files:
            zz.write(file, os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")

if __name__ == "__main__":
    main()
