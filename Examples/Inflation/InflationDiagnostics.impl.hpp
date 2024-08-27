/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(INFLATIONDIAGNOSTICS_HPP_)
#error "This file should only be included through InflationDiagnostics.hpp"
#endif

#ifndef INFLATIONDIAGNOSTICS_IMPL_HPP_
#define INFLATIONDIAGNOSTICS_IMPL_HPP_
#include "DimensionDefinitions.hpp"

template <class matter_t>
InflationDiagnostics<matter_t>::InflationDiagnostics(
    const matter_t a_matter, double dx, double G_Newton, int a_c_Ham,
    const Interval &a_c_Moms, int a_c_Ham_abs_terms /* defaulted*/,
    const Interval &a_c_Moms_abs_terms /*defaulted*/)
    : MatterConstraints<matter_t>(a_matter, dx, G_Newton,
                      a_c_Ham, a_c_Moms,
                      a_c_Ham_abs_terms,
                      a_c_Moms_abs_terms),
    my_matter(a_matter)
{
}

template <class matter_t>
template <class data_t>
void InflationDiagnostics<matter_t>::compute(Cell<data_t> current_cell) const
{
    // Load local vars and calculate derivs
    const auto vars = current_cell.template load_vars<InflationDiagnosticsVars>();
    const auto d1 = this->m_deriv.template diff1<InflationDiagnosticsVars>(current_cell);
    const auto d2 = this->m_deriv.template diff2<InflationDiagnosticsVars>(current_cell);

    // Inverse metric and Christoffel symbol
    const auto h_UU = TensorAlgebra::compute_inverse_sym(vars.h);
    const auto chris = TensorAlgebra::compute_christoffel(d1.h, h_UU);

    // Define quantities
    data_t rho;
    data_t sqrt_gam;
    data_t S;
    data_t rho_scaled;
    data_t S_scaled;
    data_t K_scaled;

    // Energy Momentum Tensor
    const auto emtensor = my_matter.compute_emtensor(vars, d1, h_UU, chris.ULL);

    //from NewConstraint
    sqrt_gam = pow(vars.chi, -3. / 2.);
    K_scaled = vars.K / pow(vars.chi, 3. / 2.);
    //from NewMatterConstraint
    rho = emtensor.rho;
    S = emtensor.S;
    rho_scaled = emtensor.rho / pow(vars.chi, 3. / 2.);
    S_scaled = emtensor.S / pow(vars.chi, 3. / 2.);
    //Write the constraints into the output FArrayBox
    current_cell.store_vars(sqrt_gam, c_sqrt_gam);
    current_cell.store_vars(rho, c_rho);
    current_cell.store_vars(rho_scaled, c_rho_scaled);
    current_cell.store_vars(S_scaled, c_S_scaled);
    current_cell.store_vars(K_scaled, c_K_scaled);


    // store_vars(out, current_cell);
}

#endif /* INFLATIONDIAGNOSTICS_IMPL_HPP_ */
