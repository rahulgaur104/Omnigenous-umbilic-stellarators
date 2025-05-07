
from desc import set_device
set_device("gpu")
import numpy as np

from desc.optimize import Optimizer
from desc.geometry import FourierRZCurve, FourierUmbilicCurve, FourierRZToroidalSurface
from desc.coils import FourierRZCoil
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.backend import jnp

from desc.grid import LinearGrid, Grid
from desc.equilibrium import Equilibrium, EquilibriaFamily

from desc.profiles import PowerSeriesProfile, SplineProfile

from desc.magnetic_fields import OmnigenousField
from desc.objectives import (
        ForceBalance,
        ObjectiveFunction, 
        QuasisymmetryTwoTerm,
        QuasisymmetryBoozer,
        QuasisymmetryTripleProduct,
        FixPsi, 
        FixBoundaryR,
        FixBoundaryZ,
        GenericObjective,
        FixPressure,
        FixCurrent,
        FixIota,
        get_fixed_boundary_constraints,
        RotationalTransform, 
        AspectRatio,
        Shear,
        Elongation, 
        UmbilicHighCurvature,
        ObjectiveFunction,
        FixCoilCurrent,
        FixPressure,
        LinearObjectiveFromUser,
        Omnigenity,
        EffectiveRipple,
        FixOmniBmax,
        FixOmniMap,
        MercierStability,
        )


from desc.integrals import Bounce2D

from desc.plotting import *
from matplotlib import pyplot as plt
import pdb

NFP_umbilic_factor = int(5)
restart_idx = int(0)


Lres = 12
Mres = 14
Nres = 14


L_shift = int(3)
M_shift = int(3)
N_shift = int(3)
L_well = int(2)
M_well = int(3)

mirror_ratio = 0.1
well_width = 0.1
radial_shift = 0.

m = int(2)
n = int(NFP_umbilic_factor)


################################################################
#########-----------PARAMETRIZING THE SURFACE-----------########
################################################################

minor_radius = 0.13
r1 = 0.133
r2 = 0.00
r3 = 0.0
r4 = 0.00
r5 = 0.1
r6 = 0
r7 = -0.00


# Define the parametrization of the umbilic torus
ntheta = int(300) 
nphi = int(300)

NFP = int(2) # For the surface parametrization, we don't use umbilic

phi_arr = np.linspace(0,  2 * np.pi/NFP, nphi)
t = np.linspace(0, 2 * np.pi, ntheta)

phi, t = np.meshgrid(phi_arr,t)
phi = phi.flatten()
t = t.flatten()

data = np.zeros((ntheta*nphi, 3))

# First, we parametrize the surface.
R =  1 + 2*minor_radius*np.cos(np.pi/(2*n)) * np.cos((t + np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n)/2 + (m*NFP*phi + r1*np.sin(NFP*phi) + r4*np.sin(2*phi))/n ) - minor_radius*np.cos(np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n + (m*NFP*phi + r1*np.sin(NFP*phi) + r4*np.sin(2*NFP*phi))/n) + r2*np.cos(NFP*phi) + r7*np.cos(2*NFP*phi)
Z = 2*minor_radius*np.cos(np.pi/(2*n)) * np.sin((t + np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n)/2 + (m*NFP*phi + r5*np.sin(NFP*phi))/n) - minor_radius*np.sin(np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n + (m*NFP*phi + r5*np.sin(phi))/n) + r3*np.sin(2*phi) + r6*np.sin(phi)
 
R_axis = 1 + r2*np.cos(NFP*phi) + r7*np.cos(2*NFP*phi)
Z_axis = r3*np.sin(2*NFP*phi) + r6*np.sin(NFP*phi)

theta_arctan = np.arctan2(Z-Z_axis, R-R_axis)
theta_arctan = np.mod(theta_arctan, 2*jnp.pi)
arr = np.array([R, phi, Z]).T
data[:, :] = arr 
theta = t

surface = FourierRZToroidalSurface.from_values(data, theta_arctan, M=Mres, N=Nres, NFP=NFP, sym=True)
print("surface parametrized! \n")

################################################################
#########-----------PARAMETRIZING THE CURVE------------#########
################################################################

phi_arr = np.linspace(0, 2 * np.pi * NFP_umbilic_factor, nphi, endpoint=False)
phi = phi_arr.flatten()
t = 0

R =  1 + 2*minor_radius*np.cos(np.pi/(2*n)) * np.cos((t + np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n)/2 + (m*NFP*phi + r1*np.sin(NFP*phi) + r4*np.sin(2*NFP*phi))/n ) - minor_radius*np.cos(np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n + (m*NFP*phi + r1*np.sin(NFP*phi) + r4*np.sin(2*NFP*phi))/n) + r2*np.cos(NFP*phi) + r7*np.cos(2*NFP*phi)
Z = 2*minor_radius*np.cos(np.pi/(2*n)) * np.sin((t + np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n)/2 + (m*NFP*phi + r5*np.sin(NFP*phi))/n) - minor_radius*np.sin(np.pi*(2*np.floor(n*t/(2*np.pi)) + 1)/n + (m*NFP*phi + r5*np.sin(NFP*phi))/n) + r3*np.sin(2*NFP*phi) + r6*np.sin(NFP*phi)

R_axis = 1 + r2*np.cos(NFP*phi) + r7*np.cos(2*NFP*phi)
Z_axis = r3*np.sin(2*NFP*phi) + r6*np.sin(NFP*phi)

theta_arctan = np.arctan2(Z-Z_axis, R-R_axis)
theta_arctan = np.mod(theta_arctan, 2*jnp.pi)

data_curve = np.zeros((nphi, 2))

theta_arctan[0] = 0.
A = n * np.unwrap(theta_arctan) - m * NFP*phi

arr = np.array([phi, A]).T
data_curve[:, :] = arr 

curve = FourierUmbilicCurve.from_values(coords=jnp.array(data_curve), N=16, NFP=1, NFP_umbilic_factor=NFP_umbilic_factor, sym="sin")
curve0 = curve.copy()

#pdb.set_trace()
print("A_n = ", curve.UC_n)
print("curve parametrized!")

# Next, we calculate the initial equilibrium
NL = 11
grid = LinearGrid(L = NL, M=0, N=0)
rho = np.linspace(0, 1, NL+1)


current_profile = SplineProfile(np.zeros((NL+1, )), knots=rho)
pressure_profile = SplineProfile(np.zeros((NL+1, )), knots=rho)


# Next, we calculate the initial equilibrium
NL = 11
grid = LinearGrid(L = NL, M=0, N=0)
rho = np.linspace(0, 1, NL+1)


iota_profile = PowerSeriesProfile.from_values(rho, 0.755 + 0.042*rho**2, order=2*Lres, sym=True)
pressure_profile = PowerSeriesProfile([5e3, 0., -5e3, 0.])


eq_nfp1 = Equilibrium(
    L=Lres,  # radial resolution
    M=Mres,  # poloidal resolution
    N=Nres,  # toroidal resolution
    surface=surface,
    pressure=pressure_profile,
    #current=current_profile,
    iota=iota_profile,
    Psi=np.pi*(minor_radius)**2,  # total flux, in Webers
)

objective = ObjectiveFunction(ForceBalance(eq=eq_nfp1))

constraints = (
        FixPressure(eq=eq_nfp1),
        FixIota(eq=eq_nfp1),
        FixPsi(eq=eq_nfp1),
        FixBoundaryR(eq=eq_nfp1),
        FixBoundaryZ(eq=eq_nfp1),
        )

try:
    eq_nfp1 = Equilibrium.load(f"eq_limiota_m{m}_n{n}_L{Lres}_M{Mres}_N{Nres}_NFP{NFP}_QA_finite_beta_neg_shear_init.h5")
except:
    eq_nfp1, _ = eq_nfp1.solve(objective=objective, constraints=constraints, ftol=1e-3, xtol=1e-5, gtol=1e-10, maxiter=71, verbose=3, copy=True)
    #eq_nfp1.save(f"eq_limiota_m{m}_n{n}_L{Lres}_M{Mres}_N{Nres}_NFP{NFP}_QA_finite_beta_pos_shear_init.h5")
    eq_nfp1.save(f"eq_limiota_m{m}_n{n}_L{Lres}_M{Mres}_N{Nres}_NFP{NFP}_QA_finite_beta_neg_shear_init.h5")
    print("Equilibrium saved ! \n")




eq = eq_nfp1

# Finally, we perform optimization on the eq, curve thing
eq_weights = [1e0, 1e2, 1e4, 1e5, 1e6, 1e7]

if restart_idx == 0:
    iter_idx = 1
else:
    opt_indices = opt_indices[restart_idx-2:]
    

#######################################################################
############------------MOVING THE CURVE ONLY-------------#############
#######################################################################

phi1 = np.linspace(0, 2 * np.pi * NFP_umbilic_factor, nphi)
curve_grid = LinearGrid(zeta = phi1, NFP_umbilic_factor=NFP_umbilic_factor)

objective = ObjectiveFunction(
    (
    UmbilicHighCurvature(eq, curve, bounds=(-80, -35)),
    GenericObjective(f="UC", thing=curve, grid=curve_grid, bounds = (-1.0*np.pi, 1.0*np.pi), weight=3e2),
    )
)
constraints = (
    ForceBalance(eq=eq),
    FixPressure(eq=eq),
    #FixCurrent(eq=eq),
    FixIota(eq=eq),
    FixPsi(eq=eq),
    FixBoundaryR(eq=eq),
    FixBoundaryZ(eq=eq),
)
optimizer = Optimizer("proximal-lsq-exact")
(eq_new, curve_opt), _ = optimizer.optimize(
    things=(eq, curve), objective=objective, ftol=1e-3, constraints=constraints, maxiter=10, verbose=2, copy=True
)

eq = eq_new.copy()
curve = curve_opt.copy()



omni_type = "OP"

if omni_type == "OT":
    helicity = (1, 0)
elif omni_type == "OH":
    helicity = (1, eq.NFP)
else:
    helicity = (0, eq.NFP)

L_shift = 3
M_shift = 3
N_shift = 3
L_well = 2
M_well = 3


ripple_weights = [5e2, 1e3, 9e3, 1.5e4]
omni_weights = [60, 50, 32, 2]
#omni_weights = [5, 5, 5, 2]

def mirrorRatio(params):
    B_lm = params["B_lm"]
    f = jnp.array([B_lm[0] - B_lm[field.M_B],  B_lm[field.M_B - 1] - B_lm[-1], ])
    return f

field = OmnigenousField(
        L_B=L_well,  # radial resolution of B_lm parameters
        M_B=M_well,  # number of spline knots on each flux surface
        L_x=L_shift,  # radial resolution of x_lmn parameters
        M_x=M_shift,  # eta resolution of x_lmn parameters
        N_x=N_shift,  # alpha resolution of x_lmn parameters
        NFP=eq.NFP,  # number of field periods; should always be equal to Equilibrium.NFP
        #helicity=(1, eq.NFP),  # helicity for poloidally closed |B| contours
        #helicity=(eq.NFP, 0),  # helicity for poloidally closed |B| contours
	helicity=helicity,
	B_lm = np.array([[1-mirror_ratio, 1+well_width, 1+mirror_ratio], [radial_shift, radial_shift, radial_shift], [0., 0., 0,]]).flatten()
        )

#######################################################################
##########-----------MOVING THE CURVE AND SURFACE -----------##########
#######################################################################

#for k in np.arange(2, 14, 2):
idx_arrays = np.array([2, 4, 4, 6, 8, 10, 12, 14])
for k in idx_arrays:

    try:
        ripple_weight = ripple_weights[int(k/2)-1]
    except:
        ripple_weight = ripple_weights[-1]

    try:
        omni_weight = omni_weights[int(k/2)-1]
    except:
        omni_weight = omni_weights[-1]

    eq_grids_omni = {}
    field_grids_omni = {}
    objs_omni = {}

    modes_R = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    modes_Z = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) >  k, :
    ]
    
    
    RotationalTransform_grid = LinearGrid(M =eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([1.0]), sym=True)
    RotationalTransform_grid2 = LinearGrid(M =eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.05]), sym=True)
    Elongation_grid = LinearGrid(M =eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([1.0]), sym=True)
    iota_values = eq.compute("iota", grid = LinearGrid(L=200))["iota"]
    
    
    phi1 = phi.copy()
    
    curve_grid = LinearGrid(zeta = phi1, NFP_umbilic_factor=NFP_umbilic_factor)
    print("iotamin, iotamax", np.min(iota_values), np.max(iota_values), np.min(np.abs(iota_values)))
    iota_sign = np.sign(iota_values[-1])
    print(iota_sign)    

    surfaces_omni = [0.25, 0.6, 0.96]
    well_weight = 1

    for rho in surfaces_omni:
        eq_grids_omni[rho] = LinearGrid(rho=rho, M = 3 * eq.M, N= 3 * eq.N, NFP=eq.NFP, sym=False)
        #field_grids_omni[rho] = LinearGrid(rho=rho, theta=20, zeta=12, NFP=field.NFP, sym=False)
        field_grids_omni[rho] = LinearGrid(rho=rho, zeta=np.linspace(0.8/field.NFP, (2*np.pi-0.8)/field.NFP, 12), theta=16, NFP=field.NFP, sym=False)
        objs_omni[rho] = Omnigenity(
            field=field,
            eq=eq,
            eq_grid=eq_grids_omni[rho],
            field_grid=field_grids_omni[rho],
            eta_weight=well_weight,
            weight = omni_weight, 
            #deriv_mode = "fwd",
        )

    surfaces_ripple = [0.05, 0.55, 0.96]
    grid_ripple = LinearGrid(rho=surfaces_ripple, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    num_transit = 15
    obj_ripple = EffectiveRipple(
        eq,
        grid=grid_ripple,
        X=16,
        Y=32,
        num_transit=num_transit,
        num_quad=13,
        num_pitch=30,
        weight = ripple_weight,
        deriv_mode="rev"
        )


    Mercier_grid = LinearGrid(M =int(eq.M), N = int(eq.N), rho=np.linspace(0.2, 1, 5), NFP=eq.NFP, sym=True, axis=False)

    objective = ObjectiveFunction(
        (
        ForceBalance(eq=eq, weight=4e5),
        UmbilicHighCurvature(eq=eq, curve=curve, bounds=(-88, -25), weight=5),
        AspectRatio(eq=eq, bounds=(4, 12), weight=10),
        GenericObjective(f="UC", thing=curve, grid=curve_grid, bounds = (-1.0*np.pi, 1.0*np.pi), weight=3e2),
        GenericObjective(f="iota vacuum", thing=eq, grid=RotationalTransform_grid, bounds = (0.15, 0.5), weight=2e2),
        obj_ripple,
        ) + tuple(objs_omni.values())
    )
    
    constraints = (
        ForceBalance(eq=eq),
        FixPressure(eq=eq),
        FixIota(eq=eq),
        FixPsi(eq=eq),
        FixBoundaryR(eq=eq, modes=modes_R),
        FixBoundaryZ(eq=eq, modes=modes_Z),
        FixOmniBmax(field=field),
        FixOmniMap(field=field, indices=np.where(field.x_basis.modes[:, 1] == 0)[0]),
    	# fix the mirror ratio on the magnetic axis
        LinearObjectiveFromUser(mirrorRatio, field, target=[0.7, 1.3]),
    )
    
    optimizer = Optimizer("proximal-lsq-exact")
    
    (eq_new, curve_opt, field_opt), _ = optimizer.optimize(
        things=(eq, curve, field), objective=objective, constraints=constraints, verbose=2, maxiter=42, ftol=1e-4, copy=True
    )

    L0 = 200
    rho = np.linspace(0, 1, L0+1)
    grid = LinearGrid(L=L0, M = eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=True)
    data_keys = ["iota", "D_Mercier", "magnetic well"]
    data = eq.compute(data_keys, grid=grid)
    
    iota = grid.compress(data["iota"])
    
    print("D_Mercier=", grid.compress(data["D_Mercier"]));
    #print("iota=", iota)
    print("beta=", eq.compute("<beta>_vol")["<beta>_vol"])  
    
    num_transit = 20
    rho_ripple = np.linspace(0.05, 1, 11)
    grid_ripple_new = LinearGrid(rho=rho_ripple, M = eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=True)
    data = eq.compute(
        "effective ripple",
        grid=grid_ripple_new,
        theta = Bounce2D.compute_theta(eq, 32, 64, rho=rho_ripple),
        Y_B=128,
        num_transit=num_transit,
        num_well=20 * num_transit,
    )
    print("Effective ripple = ", grid_ripple_new.compress(data["effective ripple"]))

    eq = eq_new.copy()
    curve = curve_opt.copy()
    field = field_opt.copy()

    eq_new.save(f"eq_omni_m{m}_NFP{NFP_umbilic_factor}_{k}.h5")
    curve_opt.save(f"curve_omni_m{m}_NFP{NFP_umbilic_factor}_{k}.h5")

    # Plotting optimized eq + curve combo
    fig = plot_3d(eq_new,"curvature_k2_rho")
    phi_arr1 = np.linspace(0, 2 * np.pi * NFP_umbilic_factor, nphi)
    phi1 = phi_arr1.flatten()

    data_curve_opt = curve_opt.compute(["UC"], grid = LinearGrid(zeta = phi1, NFP_umbilic_factor=NFP_umbilic_factor), override_grid=False)
    theta1 = (data_curve_opt["UC"] - m * NFP * phi1)/NFP_umbilic_factor
    custom_grid = Grid(jnp.array([jnp.ones_like(phi1), theta1, phi1]).T)
    curve_data = eq_new.compute(["R", "Z"], grid=custom_grid)
    R1 = curve_data["R"]
    Z1 = curve_data["Z"]
    data_curve_opt1 = np.zeros((len(phi1), 3))

    arr1 = np.array([R1, phi1, Z1]).T
    data_curve_opt1[:, :] = arr1

    fig.add_scatter3d(
    x=R1*np.cos(phi1),
    y=R1*np.sin(phi1),
    z=Z1,
    marker=dict(
    size=0,
    opacity=0,
    ),
    line=dict(
    color="black",
    width=10,
    dash="solid",
    ),
    showlegend=False,)


    #curve_opt1 = FourierRZCurve.from_values(coords=jnp.array(data_curve_opt1), N=21, NFP=1, NFP_umbilic_factor=NFP_umbilic_factor)
    #plot_coils(curve_opt1, fig=fig,grid=custom_grid)

    fig.write_html(f"eq_and_coil_opt_omni_m{m}_NFP{NFP_umbilic_factor}_{k}.html")
    plt.close()

    fig, ax = plot_section(eq_new, name="|F|",norm_F=True, log=True)
    plt.savefig(f"F_norm_omni_m{m}_NFP{NFP_umbilic_factor}_{k}.png", dpi=400)
    plt.close()

    fig, ax = plot_comparison(eqs=[eq])
    plt.savefig(f"Xsection_omni_m{m}_NFP{NFP_umbilic_factor}_{k}.png", dpi=400)

    fig, ax = plot_boozer_surface(eq, rho=1)
    plt.savefig(f"Boozer_eq_omni_m{m}_NFP{NFP_umbilic_factor}_{k}.png", dpi=400)
    plt.close()

    fig, ax = plot_boozer_surface(eq, rho=0.5)
    plt.savefig(f"Boozer_eq_omni_0p5_m{m}_NFP{NFP_umbilic_factor}_{k}.png", dpi=400)
    plt.close()




