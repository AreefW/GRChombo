/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef INITIALK_HPP
#define INITIALK_HPP

#include "CCZ4Geometry.hpp"
#include "Cell.hpp"
#include "FourthOrderDerivatives.hpp"
#include "Interval.H"
#include "MatterCCZ4.hpp"
#include "simd.hpp"
#include "DimensionDefinitions.hpp"

//! Calculates the EM tensor and then saves the ones specified in the
//! constructor on the grid
template <class matter_t> class InitialK
{
  public:
    template <class data_t>
    using Vars = typename MatterCCZ4<matter_t>::template Vars<data_t>;

    //! Constructor
    InitialK(const matter_t &a_matter, const double dx, const int a_c_rho = -1,
             const Interval a_c_Si = Interval(),
             const Interval a_c_Sij = Interval());

    template <class data_t> void compute(Cell<data_t> current_cell) const;

  protected:
    const matter_t &m_matter;
    FourthOrderDerivatives m_deriv;
    const int m_c_rho;      // var enum for the energy density
    const Interval m_c_Si;  // Interval of var enums for the momentum density
    const Interval m_c_Sij; // Interval of var enums for the spatial
                            // stress-energy density
};

template <class matter_t>
InitialK<matter_t>::InitialK(const matter_t &a_matter, const double dx,
                             const int a_c_rho, const Interval a_c_Si,
                             const Interval a_c_Sij)
    : m_matter(a_matter), m_deriv(dx), m_c_rho(a_c_rho), m_c_Si(a_c_Si),
      m_c_Sij(a_c_Sij)
{
    // if (m_c_Si.size() != 0)
    // {
    //     // Si is a vector
    //     CH_assert(m_c_Si.size() == DEFAULT_TENSOR_DIM);
    // }

    // if (m_c_Sij.size() != 0)
    // {
    //     // Sij is a symmetric tensor
    //     CH_assert(m_c_Sij.size() ==
    //               DEFAULT_TENSOR_DIM * (DEFAULT_TENSOR_DIM + 1) / 2);
    //}
}

template <class matter_t>
template <class data_t>
void InitialK<matter_t>::compute(Cell<data_t> current_cell) const
{
    const auto vars = current_cell.template load_vars<Vars>();
    const auto d1 = m_deriv.template diff1<Vars>(current_cell);

    using namespace TensorAlgebra;

    const auto h_UU = compute_inverse_sym(vars.h);
    const auto chris = compute_christoffel(d1.h, h_UU);

    const auto emtensor = m_matter.compute_emtensor(vars, d1, h_UU, chris.ULL);

    data_t K = -sqrt(24*M_PI*emtensor.rho);   
    current_cell.store_vars(K, c_K);
    
}


#endif /* InitialK_HPP */
