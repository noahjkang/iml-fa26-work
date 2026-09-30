import os

file_path = "verification/part2_integrator.jl"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "final_set[1]" in line:
        new_lines.append("    rect = overapproximate(final_set, Hyperrectangle)\n")
        new_lines.append('    println("a  ∈ [", low(rect)[1], ", ", high(rect)[1], "]")\n')
    elif "final_set[2]" in line:
        new_lines.append('    println("a'' ∈ [", low(rect)[2], ", ", high(rect)[2], "]")\n')
    elif "final_set[3]" in line:
        new_lines.append('    println("b  ∈ [", low(rect)[3], ", ", high(rect)[3], "]")\n')
    elif "final_set[4]" in line:
        new_lines.append('    println("b'' ∈ [", low(rect)[4], ", ", high(rect)[4], "]")\n')
    elif "final_set[5]" in line:
        new_lines.append('    println("f  ∈ [", low(rect)[5], ", ", high(rect)[5], "]")\n')
    elif "final_set[6]" in line:
        new_lines.append('    println("f'' ∈ [", low(rect)[6], ", ", high(rect)[6], "]")\n')
    else:
        new_lines.append(line)

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
