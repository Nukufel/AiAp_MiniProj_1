import subprocess

project_dir = "./"

print(f"Running blue ")

subprocess.run(["blue", project_dir], check=True)

print("blue done.")