using ReadVTK, Interpolations, JLD2, TypedTables, CairoMakie, ColorSchemes

# these are relative to 1 CSS px (matches examples/figures.jl house style)
inch = 96
pt = 4/3
cm = inch / 2.54

H, rings = 1, 8

# --- flow slice (N=256, θ=π/8 azimuthal cut) ---
Nflow = 256
R = 2Nflow / 3

vtk = VTKFile("vtk_data/kirigami_N$(Nflow)_H$(H)_rings$(rings)_000000.vti")
pd = get_point_data(vtk)
ω = get_data_reshaped(pd["ω"])   # (3, nx, ny, nz)
d = get_data_reshaped(pd["d"])   # (nx, ny, nz) signed distance to body

nx, ny, nz = size(ω, 2), size(ω, 3), size(ω, 4)

# index 1 is an unpopulated ghost layer (identically zero) along every axis - drop it
xs, ys, zs = 2:nx, 2:ny, 2:nz
ωy_itp = linear_interpolation((xs, ys, zs), ω[2, xs, ys, zs])
ωz_itp = linear_interpolation((xs, ys, zs), ω[3, xs, ys, zs])
d_itp = linear_interpolation((xs, ys, zs), d[xs, ys, zs])

# θ=π/8: where cos(4θ+ϕ) = 0 for both ring phases (ϕ=0,π), so every ring sits at its
# neutral radial offset instead of alternating extremes - avoids the θ=0 ring-fusing artifact
θ = π / 8
sθ, cθ = sincos(θ)

rs = 2:ny   # radial samples, index units, r=0 at rs[1]
X = [(xi - 2) / R for xi in xs]
Rr = [(ri - 2) / R for ri in rs]

ωθ = similar(X, length(xs), length(rs))
body = similar(ωθ)
for (ix, xi) in enumerate(xs), (ir, ri) in enumerate(rs)
    yc, zc = 2 + (ri - 2) * cθ, 2 + (ri - 2) * sθ
    ωθ[ix, ir] = -sθ * ωy_itp(xi, yc, zc) + cθ * ωz_itp(xi, yc, zc)
    body[ix, ir] = d_itp(xi, yc, zc)
end

clim = 0.2
thresh = 0.02  # mask near-zero band: mostly far-field solver noise, not real structure
flowdata = clamp.(ωθ, -clim, clim)
flowdata[abs.(ωθ).<thresh] .= NaN

# --- convergence sweep (N=64..256) ---
Ns = (64, 96, 128, 192, 256)
data = Dict(N => load_object("kirigami_N$(N)_H$(H)_rings$(rings)_hist.jld2") for N in Ns)
Nref = last(Ns)
Cd_ref = data[Nref].Cd
coarse = collect(Ns[1:end-1])
err = [sqrt(sum((data[N].Cd .- Cd_ref) .^ 2) / sum(Cd_ref .^ 2)) for N in coarse]

# --- figure: flow on top, Cd traces + convergence below ---
f = Figure(size=(22cm, 14cm), figure_padding=6, fontsize=9pt)

axflow = Axis(f[2, 1:4], aspect=DataAspect(), xlabel="X/R", ylabel="r/R (θ=π/8)")
co = contourf!(axflow, X, Rr, flowdata, levels=range(-clim, clim, 17), colormap=:RdBu)
contour!(axflow, X, Rr, body, levels=[0], color=:black, linewidth=1.5)
Colorbar(f[1, 1:4], co, vertical=false, label="Azimuthal vorticity ω_θ", labelsize=12, width=Relative(0.6))
xlims!(axflow, extrema(X)...)
ylims!(axflow, extrema(Rr)...)

# columns 1 and 4 are narrow spacers so the bottom pair's combined width matches axflow above
ax1 = Axis(f[3, 2], xlabel="Convective time", ylabel="Drag coefficient")
colors = get(ColorSchemes.Blues, range(0.35, 1.0, length=length(Ns)))
for (c, N) in zip(colors, Ns)
    lines!(ax1, data[N].t, data[N].Cd, color=c, linewidth=2, label="R=$(round(Int, 2N/3))")
end
axislegend(ax1, position=:lt, rowgap=-6, patchlabelgap=2, patchsize=(16, 20),
    padding=(10, 10, -2, -2), framevisible=false)
xlims!(ax1, 0, 3)

Rcoarse = round.(Int, 2 .* coarse ./ 3)
ax2 = Axis(f[3, 3], xlabel="Grid resolution R", ylabel="log₁₀(L2 error)",
    xscale=log10, xticks=(Rcoarse, string.(Rcoarse)), yticks=[-1, -2])
scatter!(ax2, Rcoarse, log10.(err), color=:black, markersize=10)
lines!(ax2, Rcoarse, log10.(err), color=:black, linewidth=1)
Nref_line = range(Rcoarse[1], Rcoarse[end], 50)
o1 = log10.(err[1] .* (Rcoarse[1] ./ Nref_line) .^ 1)
o2 = log10.(err[1] .* (Rcoarse[1] ./ Nref_line) .^ 2)
lines!(ax2, Nref_line, o1, color=:gray, linestyle=:dot, linewidth=1, label="1st order")
lines!(ax2, Nref_line, o2, color=:gray, linestyle=:dash, linewidth=1, label="2nd order")
axislegend(ax2, position=:lb, rowgap=-6, patchlabelgap=2, patchsize=(16, 20),
    padding=(10, 10, -2, -2), framevisible=false)
ylims!(ax2, -2, maximum(log10.(err)) + 0.05)

rowgap!(f.layout, 10)
colgap!(f.layout, 10)
colgap!(f.layout, 1, 0)  # no gap between left spacer and ax1
colgap!(f.layout, 3, 0)  # no gap between ax2 and right spacer
rowsize!(f.layout, 2, Auto(1.0))  # bigger flow plot
rowsize!(f.layout, 3, Auto(0.625))  # bottom two plots: 50% -> 62.5% (+25%) of their original size
colsize!(f.layout, 1, Fixed(0.34cm))  # left spacer: aligns ax1 with axflow's left edge
colsize!(f.layout, 4, Fixed(1.25cm))  # right spacer: aligns ax2 with axflow's right edge

save("tex/fig/kirigami_convergence.png", f, px_per_unit=300/inch)
f
