# read main.leo file
import os

with open(os.path.join(os.getcwd(), "zkml-research", "zkml-web-app", 'main.leo'), 'r') as f:
    lines = f.readlines()

lines_changed = []

for line in lines:
    # replace i64 in the line with ""
    line = line.replace("i64", "")

    if("if " in line):
        # replace if with if(
        line = line.replace("if ", "if(")
        line = line.replace(" {", "){")

    lines_changed.append(line)

# store the lines in a new file
with open(os.path.join(os.getcwd(), "zkml-research", "zkml-web-app", 'main.js'), 'w') as f:
    f.writelines(lines_changed)

a = []