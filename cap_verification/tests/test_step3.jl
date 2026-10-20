using IntervalArithmetic

R = 20.0
box_a  = interval(9.516773350254843, 9.532952062102746)
box_ap = interval(0.4430850844331388, 0.4638037907813664)
box_b  = interval(12.313009418666901, 12.334303182397065)
box_bp = interval(0.5771451420743623, 0.6039848800258709)
box_fp = interval(-20.928832927960276, -20.890197414543056)

direct = box_ap * box_fp + box_a
println("Direct: ", direct)

w = box_fp + R
z_a = box_a - R * box_ap
centered = z_a + box_ap * w
println("Centered: ", centered)
println("z_a = ", z_a)
