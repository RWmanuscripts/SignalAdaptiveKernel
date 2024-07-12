
import LazyGPR as LGP

import VisualizationBag as VIZ
import PythonPlot as PLT
fig_num = 1
PLT.close("all")

T = Float64

fig_size = (5, 1)
dpi = 300

# ## Visualize RQ curve
b_x = convert(T, 5)
M = floor(Int, b_x)
L = M # must be an even positive integer. The larger the flatter.
if isodd(L)
    L = L + 1
end
x0, y0 = convert(T, 0.8*M), 1 + convert(T, 0.5)
s = LGP.AdjustmentMap(x0, y0, b_x, L)

# plot.
viz_bound = 0.9
u_range = LinRange(-viz_bound*M, viz_bound*M, 1000)
g = uu->LGP.evalsmap(uu, s)


PLT.figure(fig_num; figsize = fig_size, dpi = dpi)
fig_num += 1
PLT.plot(u_range, g.(u_range))
PLT.xlabel("τ")
PLT.ylabel("s(τ)")
PLT.show()

# verify implementation with formula.
s2 = (tt,aa,bb)->(-(aa*tt^L+bb^L)/(tt^L-bb^L))
@assert s2(u_range[end], s.a, s.b) - g(u_range[end]) < 1e-5 # should be practically zero.

nothing