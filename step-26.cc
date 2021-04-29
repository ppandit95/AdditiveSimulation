/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>


namespace Step26
{
  using namespace dealii;


  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation();
    void run();

  private:
    void create_coarse_grid();
    bool cell_is_in_metal_domain();
    bool cell_is_in_void_domain();
    void set_active_fe_indice();
    void setup_system();
    void store_old_vector();//Need to define a map to store old solution
    void transfer_old_vector();//Uses the map filled by store_old_vector()
    void assemble_system();
    void solve_time_step();
    void output_results() const;
    void refine_mesh (const unsigned int min_grid_level,
                      const unsigned int max_grid_level);

    Triangulation<dim>   triangulation;
    hp::FECollection<dim>fe_collection;
    hp::DoFHandler<dim>  dof_handler;
    hp::QCollection<dim> quadrature_collection;

    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       old_solution;
    Vector<double>       system_rhs;

    double               time;
    double               time_step;
    unsigned int         timestep_number;

    const double         theta;

    const double 		 edge_length;
    const double		 layerThickness;
    const double 		 number_layer;
  };




  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide ()
      :
      Function<dim>(),
      period (0.2)
    {}

    virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;

  private:
    const double period;
  };



  template <int dim>
  double RightHandSide<dim>::value (const Point<dim> &p,
                                    const unsigned int component) const
  {
    (void) component;
    Assert (component == 0, ExcIndexRange(component, 0, 1));
    Assert (dim == 2, ExcNotImplemented());

    const double time = this->get_time();
    const double point_within_period = (time/period - std::floor(time/period));

    if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
      {
        if ((p[0] > 0.5) && (p[1] > -0.5))
          return 1;
        else
          return 0;
      }
    else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
      {
        if ((p[0] > -0.5) && (p[1] > 0.5))
          return 1;
        else
          return 0;
      }
    else
      return 0;
  }



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value (const Point<dim>  &p,
                          const unsigned int component = 0) const;
  };



  template <int dim>
  double BoundaryValues<dim>::value (const Point<dim> &/*p*/,
                                     const unsigned int component) const
  {
    (void) component;
    Assert (component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }



  template <int dim>
  HeatEquation<dim>::HeatEquation ()
    :
    dof_handler(triangulation),
    time (0.0),
    time_step(1. / 500),
    timestep_number (0),
    theta(0.5),
    edge_length(1.0),
	layerThickness(0.05),
	number_layer(20)
  {
	  fe_collection.push_back(FE_Q<dim>(1));
	  fe_collection.push_back(FE_Nothing<dim>());

	  quadrature_collection.push_back(QGauss<dim>(2));
	  quadrature_collection.push_back(FE_Nothing<dim>());
  }

  template<int dim>
  	  void HeatEquation<dim>::create_coarse_grid(){
	  //Creation of two points
	  Point<dim> p1;
	  Point<dim> p2;

	  //Writing coordinates to p1 and p2
	  for(unsigned int n=0;n<dim;n++){
		  p1[n] = 0;
		  p2[n] = edge_length;
	  }

	  p2[dim-1] = layerThickness*number_layer;

	  //Generate a parallelopiped with a [p1 p2] diagonal
	  GridGenerator::hyper_rectangle(triangulation,p1,p2);
  }


  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
	//DOF distribution using proper finite element
	dof_handler.distribute_dofs(fe_collection);

	//Creation of the constraints to handle hanging nodes
	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	constraints.close();

	//Creation of the sparsity pattern for the matrices
	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, true);
	sparsity_pattern.copy_from(dsp);

	mass_matrix.reinit(sparsity_pattern);
	laplace_matrix.reinit(sparsity_pattern);
	system_matrix.reinit(sparsity_pattern);

	//Mass Matrix Computation
	MatrixCreator::create_mass_matrix(dof_handler,quadrature_collection,mass_matrix,(const Function<dim> *)0,constraints);

	//Stiffness Matrix Computation
	MatrixCreator::create_laplace_matrix(dof_handler, quadrature_collection, laplace_matrix, (const Function<dim> *)0,constraints);

	//Memory Allocation for the RHS vector
	system_rhs.reinit(dof_handler.n_dofs());

	//Memory Allocation for solution vector
	solution.reinit(dof_handler.n_dofs());
	old_solution.reinit(dof_handler.n_dofs());
  }


  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs,
             preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step()
              << " CG iterations." << std::endl;
  }



  template <int dim>
  void HeatEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");

    data_out.build_patches();

    const std::string filename = "solution-"
                                 + Utilities::int_to_string(timestep_number, 3) +
                                 ".vtk";
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
  }


  template <int dim>
  void HeatEquation<dim>::refine_mesh (const unsigned int min_grid_level,
                                       const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(fe.degree+1),
                                        typename FunctionMap<dim>::type(),
                                        solution,
                                        estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
                                                       estimated_error_per_cell,
                                                       0.6, 0.4);

    if (triangulation.n_levels() > max_grid_level)
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active(max_grid_level);
           cell != triangulation.end(); ++cell)
        cell->clear_refine_flag ();
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level); ++cell)
      cell->clear_coarsen_flag ();

    SolutionTransfer<dim> solution_trans(dof_handler);

    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement ();
    setup_system ();

    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute (solution);
  }



  template <int dim>
  void HeatEquation<dim>::run()
  {
    const unsigned int initial_global_refinement = 2;
    const unsigned int n_adaptive_pre_refinement_steps = 4;

    create_coarse_grid();
    triangulation.refine_global (initial_global_refinement);

    setup_system();

    unsigned int pre_refinement_step = 0;

    Vector<double> tmp;
    Vector<double> forcing_terms;

start_time_iteration:

    tmp.reinit (solution.size());
    forcing_terms.reinit (solution.size());


    VectorTools::interpolate(dof_handler,
                             Functions::ZeroFunction<dim>(),
                             old_solution);
    solution = old_solution;

    output_results();

    while (time <= 0.5)
      {
        time += time_step;
        ++timestep_number;

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        mass_matrix.vmult(system_rhs, old_solution);

        laplace_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1 - theta) * time_step, tmp);

        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree+1),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= time_step * theta;

        rhs_function.set_time(time - time_step);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree+1),
                                            rhs_function,
                                            tmp);

        forcing_terms.add(time_step * (1 - theta), tmp);

        system_rhs += forcing_terms;

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * time_step, laplace_matrix);

        constraints.condense (system_matrix, system_rhs);

        {
          BoundaryValues<dim> boundary_values_function;
          boundary_values_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_function,
                                                   boundary_values);

          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        }

        solve_time_step();

        output_results();

        if ((timestep_number == 1) &&
            (pre_refinement_step < n_adaptive_pre_refinement_steps))
          {
            refine_mesh (initial_global_refinement,
                         initial_global_refinement + n_adaptive_pre_refinement_steps);
            ++pre_refinement_step;

            tmp.reinit (solution.size());
            forcing_terms.reinit (solution.size());

            std::cout << std::endl;

            goto start_time_iteration;
          }
        else if ((timestep_number > 0) && (timestep_number % 5 == 0))
          {
            refine_mesh (initial_global_refinement,
                         initial_global_refinement + n_adaptive_pre_refinement_steps);
            tmp.reinit (solution.size());
            forcing_terms.reinit (solution.size());
          }

        old_solution = solution;
      }
  }
}


int main()
{
  try
    {
      using namespace dealii;
      using namespace Step26;

      HeatEquation<2> heat_equation_solver;
      heat_equation_solver.run();

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
