# Omnigenous Umbilic Stellarators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15355215.svg)](https://doi.org/10.5281/zenodo.15355215)


This repository contains all the data and analysis files used to generate and analyze omnigenous umbilic stellarators in this [preprint](https://arxiv.org/abs/2505.04211).
The only files that have been removed are four large data files (> 100 MB) such as output from the FIELDLINES code and mgrid files
which can be generated by running FIELDLINES and DESC.

The structure of files and directories is shown below

```
├── FIELDLINES_analysis
│   ├── beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5
│   ├── FIELDLINES_inputs\
│   ├── FIELDLINES_outputs\
│   ├── medium_beak_hres.nc
│   ├── mgrid_testcoil.nc
│   ├── mgrid_umbilic_DESC.nc
│   ├── mgrid_umbilic.nc
│   └── wout_umbilic.nc
├── HBT-EP_optimization
│   ├── 3D_plot1.html
│   ├── 3D_plot2.html
│   ├── beaked_medium_0.125.npz
│   ├── beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5
│   ├── beak_equilibrium_medium_beak_beta1p0.h5
│   ├── beak_equilibrium_umbilic_curve_A8_beta1p0.h5
│   ├── Boozer_contours\
│   ├── curvature_plot2.pdf
│   ├── curvature_plot2.png
│   ├── curvature_plot2.svg
│   ├── curvature_plot.pdf
│   ├── deformed_HBT_medium_bulge.svg
│   ├── extract_points_from_svg.py
│   ├── HBT-EP_iota_profile.pdf
│   ├── HBT-EP_pressure_profile.pdf
│   ├── medium_beak_optimized_coilset_2p100kA_N12_mindist0p020_PFcurrent_59kA_shorterTF.h5
│   ├── modB_3d_plot2.pdf
│   ├── modB_3d_plot2.png
│   ├── modB_3d_plot2.svg
│   ├── modB_3d_plot.pdf
│   ├── post_process_3d.py
│   ├── post_process_3d_umbilicoil_only.py
│   ├── post_process_Boozer.py
│   ├── post_process_fieldline_ridge.py
│   ├── post_process_profiles.py
│   ├── post_process_Xsection.py
│   ├── Xsection_optimized2.pdf
│   ├── Xsection_optimized2.svg
│   ├── Xsection_optimized.pdf
│   ├── Xsection_optimized.png
│   └── Xsection_optimized.svg
├── HBT_reverse_current
│   ├── 3D_plot1.html
│   ├── 3D_plot2.html
│   ├── beak_beta1p0_coilset_reversed_current4.h5
│   ├── beak_equilibrium_A8_beta1p0_2100.0_increased4100_fixcur.h5
│   ├── beak_equilibrium_A8_beta1p0_2100.0_increased6100_fixcur.h5
│   ├── beak_equilibrium_A8_beta1p0_4_umbilic-8100_fixcur.h5
│   ├── beak_equilibrium_A8_beta1p0_4_umbilic-8100.h5
│   ├── beak_equilibrium_A8_beta1p0_6_umbilic-16100_fixcur.h5
│   ├── beak_equilibrium_A8_beta1p0_6_umbilic-8100_fixcur.h5
│   ├── coil_position.npy
│   ├── curvature_plot1_reversed.pdf
│   ├── curvature_plot1_reversed.svg
│   ├── curvature_plot1.svg
│   ├── iota_estimate.py
│   ├── modB_3d_plot_reversed.pdf
│   ├── modB_3d_plot_reversed.svg
│   ├── modB_3d_plot.svg
│   ├── post_process_3d.py
│   ├── post_process_3d_umbilicoil_only.py
│   ├── post_process_iota_contribution.py
│   ├── post_process_umbilic_coils_pts.py
│   ├── post_process_Xsection.py
│   ├── Xsection_optimized_reverse2.svg
│   ├── Xsection_optimized_reverse.pdf
│   └── Xsection_optimized_reverse.svg
├── HBT_umbilic_coil_current_scan
│   ├── 2100\
│   ├── 4100\
│   ├── 6100\
│   ├── beak_equilibrium_A8_beta1p0_2100.0_increased4100_fixcur.h5
│   ├── beak_equilibrium_A8_beta1p0_2100.0_increased4100.h5
│   ├── beak_equilibrium_A8_beta1p0_2100.0_increased6100_fixcur.h5
│   ├── beak_equilibrium_A8_beta1p0_2100.0_increased6100.h5
│   ├── beak_equilibrium_A8_beta1p0_2100.0_reducedFalse.h5
│   ├── Boozer_contours
│   ├── calculate_toroidal_current_sign.py
│   ├── iota_fraction_w_umbilic_current.pdf
│   ├── medium_beak_optimized_coilset_2p100kA_N12_mindist0p020_PFcurrent_59kA_shorterTF.h5
│   ├── post_process_Boozer.py
│   └── post_process_iota_contribution.py
├── OP_DB_optimized
│   ├── Boozer_contours\
│   ├── eq_NFP1_A+7.0_tor+0.1_mr+0.20_shift+0.00_wd-0.05_elong+3.00_wellwt+2.0_vacuum_final2.h5
│   ├── eq_NFP2_A+6.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+2.00_wellwt+2.0_vacuum_final2.h5
│   ├── NFP1_A+7_tor+0.05_mr+0.2_shift0.0_wd-0.05_elong+3.0_wellwt+2.0_iotashift+0.0
│   └── post_process_Boozer.py
├── pwO_calculations
│   ├── Boozer_contours\
│   ├── eq_NFP1_A+7.0_tor+0.1_mr+0.20_shift+0.00_wd-0.05_elong+3.00_wellwt+2.0_vacuum_final2.h5
│   ├── Jpar_and_Boozer_OP_DB_NFP1.pdf
│   ├── Jpar_and_Boozer_OP_DB_NFP1.svg
│   ├── Jpar_and_Boozer_OP_DB.pdf
│   ├── Jpar_and_Boozer_OP_DB.svg
│   ├── Jpar_and_Boozer_UT131.pdf
│   ├── Jpar_and_Boozer_UT131.svg
│   ├── Jpar_and_Boozer_UT225.pdf
│   ├── Jpar_and_Boozer_UT225.svg
│   ├── Jpar_OP_DB_NFP1.npz
│   ├── Jpar_OP_DB_NFP1.svg
│   ├── Jpar_OP_DB_NFP2.npz
│   ├── Jpar_OP_DB_NFP2.svg
│   ├── Jpar_UT131_rho0p5.npz
│   ├── Jpar_UT131.svg
│   ├── Jpar_UT225_rho0p75.npz
│   ├── Jpar_UT225.svg
│   ├── plot_Jpar.py
│   └── post_process_Boozer.py
├── README.md
├── US131_3D_render_Figure1
│   ├── UToL423_full.svg
│   ├── UToL423_render.pdf
│   ├── UToL423_render.png
│   ├── UToL423_render.svg
│   ├── UToL423_single_period_render2.pdf
│   ├── UToL423_single_period_render.pdf
│   ├── UToL423_single_period_render.png
│   ├── UToL423_single_period_render.svg
│   └── UToLs_data.nb
├── US131_coil_design
│   ├── 3D_magax_plot.html
│   ├── 3D_plot1.html
│   ├── coil_layout_white_bkg.svg
│   ├── coil_stage_two_optimization9.ipynb
│   ├── data_co.npy
│   ├── data_counter.npy
│   ├── eq_high-res_optimized.h5
│   ├── optimized_coilset9.h5
│   ├── paraview-utils.ipynb
│   ├── post_process_3d.py
│   ├── post_process_Bn_2d.py
│   ├── post_process_magaxis.py
│   ├── post_process_poincare.py
│   ├── save_vtk.py
│   ├── test0.svg.pvtmp.svg
│   ├── test1.svg
│   ├── test2_bkp.svg
│   ├── test2.svg
│   ├── test2.svg.pvtmp.svg
│   ├── US131_Bn_umbilic_curve.svg
│   ├── US131_coilset.pdf
│   ├── US131_collage2.pdf
│   ├── US131_collage2.svg
│   ├── US131_magax1.svg
│   ├── US131_poincare_trace.pdf
│   └── US131_poincare_trace.svg
├── US131_Figure2
│   ├── eq_limiota_m1_n3_L14_M14_N14_QA_init.h5
│   ├── normF_UToL131.pdf
│   ├── normF_UToL131.svg
│   ├── normF_UToL131_zoom.pdf
│   ├── normF_UToL131_zoom.svg
│   ├── post_process_normF_inset.py
│   └── post_process_normF.py
├── US131_optimization
│   ├── Boozer_contours\
│   ├── boundary_plot\
│   ├── driver_umbilic_ripple2.py
│   ├── driver_umbilic_ripple3.py
│   ├── driver_umbilic_ripple.py
│   ├── eq_high-res_initial.h5
│   ├── eq_high-res_optimized.h5
│   ├── fieldline_umbilicedge.svg
│   ├── iota_comparison.pdf
│   ├── iota_comparison_projection.pdf
│   ├── iota_comparison_projection.svg
│   ├── iota_comparison.svg
│   ├── post_process_3D.py
│   ├── post_process_Boozer.py
│   ├── post_process_boundary.py
│   ├── post_process_fieldline_umbiliccurve.py
│   ├── post_process_iota.py
│   ├── post_process_magaxis.py
│   ├── post_process_ripple.py
│   ├── post_process_Xsection.py
│   ├── ripple_comparison.pdf
│   ├── ripple_initial.npz
│   ├── ripple_optimized.npz
│   ├── save_VMEC_wout.py
│   ├── US131_fieldline_umbilicedge.pdf
│   ├── US131_fieldline_umbilicedge.svg
│   └── wout_umbilic_131.nc
└── US252_optimization
    ├── Boozer_contours\
    ├── boundary_plots\
    ├── curvature_3d_OP_optimized.html
    ├── curvature_optimized_3D_UT225_2.svg
    ├── curvature_optimized_3D_UT225.pdf
    ├── curvature_optimized_3D_UT225.svg
    ├── curve_omni_m2_NFP5_14.h5
    ├── driver_OP.py
    ├── eq_final_high-res.h5
    ├── eq_initial_high-res_m2_NFP5.h5
    ├── eq_omni_m2_NFP5_14.h5
    ├── fieldline_umbilicedge.svg
    ├── F_norm_omni_m2_NFP5_14.png
    ├── iota_fraction_UT225.pdf
    ├── job_runner.sl
    ├── Jpar_UT225.pdf
    ├── modB_3d_OP_initial.html
    ├── modB_3d_OP_optimized.html
    ├── post_process_3D_and_curve.py
    ├── post_process_3D.py
    ├── post_process_Boozer.py
    ├── post_process_boundary2.py
    ├── post_process_boundary.py
    ├── post_process_fieldline_umbiliccurve.py
    ├── post_process_high_curvature.py
    ├── post_process_iota_contribution.py
    ├── post_process_ripple.py
    ├── ripple_comparison_UT225.pdf
    ├── ripple_initial.npz
    ├── ripple_optimized.npz
```

