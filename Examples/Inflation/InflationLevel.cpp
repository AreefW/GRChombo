/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// General includes common to most GR problems
#include "InflationLevel.hpp"
#include "BoxLoops.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "SixthOrderDerivatives.hpp"
#include "TraceARemoval.hpp"

// For RHS update
#include "MatterCCZ4RHS.hpp"

// For constraints calculation
#include "NewMatterConstraints.hpp"

// For tag cells
#include "FixedGridsTaggingCriterion.hpp"

// Problem specific includes
#include "ComputePack.hpp"
#include "GammaCalculator.hpp"
#include "InitialScalarData.hpp"
#include "KerrBH.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"
#include "SetValue.hpp"
#include "AMRReductions.hpp"
#include "InitialK.hpp"
#include "ModifiedMovingPunctureGauge.hpp"
#include "InflationDiagnostics.hpp"

// Things to do at each advance step, after the RK4 is calculated
void InflationLevel::specificAdvance()
{
    // Enforce trace free A_ij and positive chi and alpha
    BoxLoops::loop(
        make_compute_pack(TraceARemoval(),
                          PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(
            NanCheck(m_dx, m_p.center, "NaNCheck in specific Advance"),
            m_state_new, m_state_new, EXCLUDE_GHOST_CELLS, disable_simd());
}

// Initial data for field and metric variables
void InflationLevel::initialData()
{
    CH_TIME("InflationLevel::initialData");
    if (m_verbosity)
        pout() << "InflationLevel::initialData " << m_level << endl;

    // First set everything to zero then initial conditions for scalar field -
    // here a Kerr BH and a scalar field profile
    /*
    BoxLoops::loop(
        make_compute_pack(SetValue(0.), KerrBH(m_p.kerr_params, m_dx),
                          InitialScalarData(m_p.initial_params, m_dx)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);
    */
    // Set initial condition of inflaton 
    BoxLoops::loop(
        make_compute_pack(SetValue(0.),
                          InitialScalarData(m_p.initial_params, m_dx)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx), m_state_new, m_state_new,
                   EXCLUDE_GHOST_CELLS);

    // Set initial K = -sqrt(24*pi*rho)
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    InitialK<ScalarFieldWithPotential> my_initial_K(scalar_field, m_dx);
    BoxLoops::loop(my_initial_K, m_state_new, m_state_new, EXCLUDE_GHOST_CELLS);
}

#ifdef CH_USE_HDF5
// Things to do before outputting a checkpoint file
void InflationLevel::prePlotLevel()
{
    fillAllGhosts();
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoxLoops::loop(
        MatterConstraints<ScalarFieldWithPotential>(
            scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom)),
        m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    InflationDiagnostics<ScalarFieldWithPotential> inflation_diagnostics(scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom), c_Ham_abs, Interval(c_Mom_abs, c_Mom_abs));
    BoxLoops::loop(inflation_diagnostics, m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
}
#endif

// Things to do in RHS update, at each RK4 step
void InflationLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                       const double a_time)
{
    // Enforce trace free A_ij and positive chi and alpha
    BoxLoops::loop(
        make_compute_pack(TraceARemoval(),
                          PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
        a_soln, a_soln, INCLUDE_GHOST_CELLS);

    // Calculate MatterCCZ4 right hand side with matter_t = ScalarField
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    ModifiedMovingPunctureGauge modified_moving_puncture_gauge(m_p.ccz4_params);
    modified_moving_puncture_gauge.set_K_mean(m_cosmo_amr.get_K_mean());
    if (m_p.max_spatial_derivative_order == 4)
    {
        MatterCCZ4RHS<ScalarFieldWithPotential, ModifiedMovingPunctureGauge,
                      FourthOrderDerivatives>
            my_ccz4_matter(scalar_field, m_p.ccz4_params, m_dx, m_p.sigma,
                           m_p.formulation, m_p.G_Newton);
        BoxLoops::loop(my_ccz4_matter, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
    else if (m_p.max_spatial_derivative_order == 6)
    {
        MatterCCZ4RHS<ScalarFieldWithPotential, ModifiedMovingPunctureGauge,
                      SixthOrderDerivatives>
            my_ccz4_matter(scalar_field, m_p.ccz4_params, m_dx, m_p.sigma,
                           m_p.formulation, m_p.G_Newton);
        BoxLoops::loop(my_ccz4_matter, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
}

// Things to do at ODE update, after soln + rhs
void InflationLevel::specificUpdateODE(GRLevelData &a_soln,
                                         const GRLevelData &a_rhs, Real a_dt)
{
    // Enforce trace free A_ij
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);
}

void InflationLevel::preTagCells()
{
    // we don't need any ghosts filled for the fixed grids tagging criterion
    // used here so don't fill any
}

void InflationLevel::computeTaggingCriterion(
    FArrayBox &tagging_criterion, const FArrayBox &current_state,
    const FArrayBox &current_state_diagnostics)
{
    BoxLoops::loop(
        FixedGridsTaggingCriterion(m_dx, m_level, 2.0 * m_p.L, m_p.center),
        current_state, tagging_criterion);
}
void InflationLevel::specificPostTimeStep()
{
    int min_level = 0;
    // No need to evaluate the diagnostics more frequently than every coarse
    // timestep, but must happen on every level (not just level zero or data
    // will not be populated on finer levels)
    bool calculate_diagnostics = at_level_timestep_multiple(min_level);

	bool first_step = (m_time == 0.);

    if (calculate_diagnostics)
    {
	fillAllGhosts();
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    //EMTensor<ScalarFieldWithPotential> my_emtensor(scalar_field, m_dx);
    //BoxLoops::loop(my_emtensor, m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    BoxLoops::loop(
        MatterConstraints<ScalarFieldWithPotential>(
            scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom), c_Ham_abs, Interval(c_Mom_abs, c_Mom_abs)),
        m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    InflationDiagnostics<ScalarFieldWithPotential> inflation_diagnostics(scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom), c_Ham_abs, Interval(c_Mom_abs, c_Mom_abs));
    BoxLoops::loop(inflation_diagnostics, m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);

        if (m_level == min_level)
        {
            AMRReductions<VariableType::diagnostic> amr_reductions_diag(m_cosmo_amr);
            double phys_vol = amr_reductions_diag.sum(c_sqrt_gam);
            double L2_Ham = amr_reductions_diag.norm(c_Ham);
            double L2_Mom = amr_reductions_diag.norm(c_Mom);
            double Ham_abs = amr_reductions_diag.norm(c_Ham_abs);
	        //double rho_mean = amr_reductions_diag.sum(c_rho_scaled)/phys_vol;
            pout() << "phys_vol = " << phys_vol << endl;
            //pout() << "rho mean  = " << rho_mean << endl;
            //----
            //<<<<<<<<< Change this to setter/getter form!!!!
            m_cosmo_amr.set_rho_mean(amr_reductions_diag.sum(c_rho_scaled) / phys_vol);
            //m_cosmo_amr.m_rho_mean = amr_reductions_diag.sum(c_rho_scaled) / phys_vol;
            m_cosmo_amr.set_S_mean(amr_reductions_diag.sum(c_S_scaled) / phys_vol);
            //m_cosmo_amr.m_S_mean = amr_reductions_diag.sum(c_S_scaled) / phys_vol;
            m_cosmo_amr.set_K_mean(- sqrt(3.0 * m_cosmo_amr.get_rho_mean()));
            //m_cosmo_amr.m_K_mean = - sqrt(3.0 * m_cosmo_amr.m_rho_mean);
            //pout() << "rho mean = " << m_cosmo_amr.get_rho_mean << endl;
            //pout() << "S mean = " << m_cosmo_amr.get_S_mean << endl;
            //pout() << "K mean = " << m_cosmo_amr.get_K_mean << endl;
            //----
	        AMRReductions<VariableType::evolution> amr_reductions_evo(m_cosmo_amr);
            //double chi_mean = amr_reductions_evo.norm(c_chi, 2, true);
            //double chi = amr_reductions_evo.sum(c_chi) ;
            double chi_mean = amr_reductions_evo.sum(c_chi) / phys_vol ;
            //double lapse = amr_reductions_evo.sum(c_lapse) / phys_vol ;
            //pout() << "chi = " << chi << endl;
            //pout() << "chi mean = " << chimean << endl;
            //pout() << "lapse = " << lapse << endl;
            //----
           //BoxLoops::loop(SetValue(m_cosmo_amr.m_K_mean, Interval(c_K, c_K)),
           //            m_state_new, m_state_new, INCLUDE_GHOST_CELLS);
            //----
            SmallDataIO constraints_file(m_p.data_path + "data_out",
                                         m_dt, m_time, m_restart_time,
                                         SmallDataIO::APPEND, first_step);
            constraints_file.remove_duplicate_time_data();
            if (first_step)
            {
                constraints_file.write_header_line({"L^2_Ham", "L^2_Mom", "<chi>", "<rho>", "Hab_abs"});
            }
            constraints_file.write_time_data_line({L2_Ham, L2_Mom, chi_mean, m_cosmo_amr.get_rho_mean(), Ham_abs});
        }
    }
// end write output file
}