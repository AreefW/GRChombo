/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef INFLATIONDIAGNOSTICS_HPP_
#define INFLATIONDIAGNOSTICS_HPP_

#include "CCZ4Geometry.hpp"
#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DiagnosticVariables.hpp"
#include "GRInterval.hpp"
#include "Tensor.hpp"
#include "simd.hpp"
#include <array>

#include "NewMatterConstraints.hpp"

#include "FourthOrderDerivatives.hpp"


//! Calculates all the rho components in a specific theory, which
//! are stored as diagnostics

template <class matter_t> 
class InflationDiagnostics : public MatterConstraints<matter_t>
{
  public:
    // template <class data_t>
    // using MatterVars = typename matter_t::template Vars<data_t>;

    // Inherit the variable definitions from CCZ4 + matter_t
    template <class data_t>
    using InflationDiagnosticsVars = typename MatterConstraints<matter_t>::template BSSNMatterVars<data_t>;

    //! Constructor of class InflationDiagnostics
    InflationDiagnostics(const matter_t a_matter
                      , double dx, double G_Newton,
                      int a_c_Ham, const Interval &a_c_Moms,
                      int a_c_Ham_abs_terms = -1,
                      const Interval &a_c_Moms_abs_terms = Interval());

    //! The compute member which calculates the constraints at each point in the
    //! box
    template <class data_t> void compute(Cell<data_t> current_cell) const;

  protected:
    matter_t my_matter; //!< The matter object, e.g. a scalar field
    //const std::array<double, CH_SPACEDIM> m_center; //!< The center of the grid

};

#include "InflationDiagnostics.impl.hpp"

#endif /* INFLATIONDIAGNOSTICS_HPP_ */
