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
#include <deal.II/lac/sparse_direct.h>
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
#include"point_comparison_operator.hpp"

#include <fstream>
#include <iostream>
#include <cmath>


namespace AdditiveSimulation
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
    void part_height_measure();
    bool cell_is_in_metal_domain(const typename hp::DoFHandler<dim>::cell_iterator &cell);
    bool cell_is_in_void_domain(const typename hp::DoFHandler<dim>::cell_iterator &cell);
    void set_active_fe_indice();
    void setup_system();
    void store_old_vectors();//Need to define a map to store old solution
    void transfer_old_vectors();//Uses the map filled by store_old_vector()
    void assemble_system();
    void solve_time_step();
    void output_results() const;
    void refine_mesh (const unsigned int min_grid_level,
                      const unsigned int max_grid_level);

    Triangulation<dim>   triangulation;
    hp::FECollection<dim>fe_collection;
    hp::DoFHandler<dim>  dof_handler;
    hp::QCollection<dim> quadrature_collection;
    hp::QCollection<dim-1> face_quadrature_collection;

    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> boundary_matrix;

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
    const double 		 heat_capacity;
    const double 		 heat_conductivity;
    const double 		 convection_coeff;
    const double		 Tamb;
    const double 		 LaserSpeed;
    double		 		 part_height;
    std::map<Point<dim>,double> map_old_solution;
  };

  template<int dim>
  class InitialCondition : public Function<dim>
  {
  public:
	  InitialCondition():Function<dim>(){}
	  virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double InitialCondition<dim>::value(const Point<dim> &p,const unsigned int component) const
  {
	  Assert(component == 0,ExcInternalError());
	  return 1.0;//Initial Temperature value
  }


  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide (const double lspeed)
      :
      Function<dim>(),
      period (1.0), //Time needed to complete one layer
	  LaserSpeed(lspeed)
    {}
    //void set_time(const double new_time);
    virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;

  private:
    const double period;
    const double LaserSpeed;
  };



  template <int dim>
  double RightHandSide<dim>::value (const Point<dim> &p,
                                    const unsigned int component) const
  {
    (void) component;
    Assert (component == 0, ExcIndexRange(component, 0, 1));


    const double time = this->get_time();//get the time value
    const double point_within_layer = (time/period - std::floor(time/period));//Return the x coordinate of point on which the laser has to be centered
    double limit = (1+floor(time*LaserSpeed))*0.2;//Returns the y coordinate of the part surface
    double dist = point_within_layer;
    const double tol_dist = 5e-2;
    return 1000*std::exp(-2.0*std::pow(((p[0] - dist)/tol_dist),2));
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
    return 0.68;//Temperature to impose
  }



  template <int dim>
  HeatEquation<dim>::HeatEquation ()
    :
    dof_handler(triangulation),
    time (0.0),
    time_step(0.01),
    timestep_number (0),
    theta(1.0),
    edge_length(1.0),
	layerThickness(0.2),
	number_layer(5),
	heat_capacity(0.012),
  	heat_conductivity(1.0),
	convection_coeff(0.00005),
	Tamb(1.0),
	part_height(0.0),
	LaserSpeed(1)
  {
	  fe_collection.push_back(FE_Q<dim>(1));
	  fe_collection.push_back(FE_Nothing<dim>());

	  quadrature_collection.push_back(QGauss<dim>(2));
	  quadrature_collection.push_back(QGauss<dim>(2));

	  face_quadrature_collection.push_back(QGauss<dim-1>(2));
	  face_quadrature_collection.push_back(QGauss<dim-1>(2));
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
	boundary_matrix.reinit(sparsity_pattern);

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

  template<int dim>
  void HeatEquation<dim>::assemble_system()
  {
	  Vector<double> tmp;
	  Vector<double> tmp2;
	  Vector<double> forcing_terms;

	  tmp.reinit(solution.size());
	  tmp2.reinit(solution.size());
	  forcing_terms.reinit(solution.size());

	  tmp2.add(heat_capacity,old_solution);
	  mass_matrix.vmult(system_rhs,tmp2); //rhs = c*M*T^(n-1)

	  laplace_matrix.vmult(tmp, old_solution);//tmp=K*T^(n-1)
	  system_rhs.add(-(1-theta)*time_step*heat_conductivity,tmp);//rhs = (cM - (1-theta)*tau*k)*K*T^(n-1)

	  //Computation of the forcing terms(=laser heat input)
	  RightHandSide<dim> rhs_function(LaserSpeed);

	  rhs_function.set_time(time);// t=tn
	  VectorTools::create_right_hand_side(dof_handler,quadrature_collection,rhs_function,tmp); // tmp = F^n

	  forcing_terms = tmp;// Forcing terms = F^n
	  forcing_terms *= time_step * theta;//forcing_terms = tau*theta*F^n

	  rhs_function.set_time(time-time_step);//t = t(n-1)
	  VectorTools::create_right_hand_side(dof_handler,quadrature_collection, rhs_function, tmp);//tmp = F^(n-1)
	  forcing_terms.add(time_step*(1-theta),tmp);// forcing_terms = tau*theta*F^n + tau*(1-theta)*F^(n-1)

	  system_rhs += forcing_terms; //rhs = tau*theta*F^n+tau*(1-theta)*F^(n-1)+ (c*M-(1-theta)*tau*k*K)*T^(n-1)

	  system_matrix.add(heat_capacity,mass_matrix); //A = cM
	  system_matrix.add(theta*time_step*heat_conductivity,laplace_matrix);// A = c*M+ theta*K*tau*K

	  //Applying Robin BCs

	  const unsigned int n_face_q_points = face_quadrature_collection[0].size();// quadrature points on faces
	  const unsigned int n_q_points = quadrature_collection[0].size();//quadrature points on elements

	  //Finite Element evaluated in quadrature points of a cell
	  hp::FEValues<dim> hp_fe_values(fe_collection,quadrature_collection,update_values | update_quadrature_points | update_JxW_values);

	  // Finite element evaluated in quadrature points of the faces of a cell
	  hp::FEFaceValues<dim> hp_fe_face_values(fe_collection,face_quadrature_collection,update_values | update_quadrature_points | update_JxW_values);

	  //Iteration over all the cells
	  typename hp::DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(),endc = dof_handler.end();

	  for(;cell!=endc;++cell)
	  {
		  const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
		  FullMatrix<double>   cell_matrix;
		  Vector<double>	cell_rhs;
		  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		  if(dofs_per_cell!=0) //Skip the cells which are in the void domain
		  {
			  cell_matrix.reinit(dofs_per_cell,dofs_per_cell);
			  cell_matrix = 0;
			  cell_rhs.reinit(dofs_per_cell);
			  cell_rhs = 0;

			  for(unsigned int face_number=0;face_number<GeometryInfo<dim>::faces_per_cell;++face_number)
			  {
				  //Tests to select the cell faces which belong to the Robin Boundary
				  if(cell->face(face_number)->at_boundary() && (cell->face(face_number)->boundary_id()==0))
				  {
					  //Term to be added in the RHS
					  hp_fe_face_values.reinit(cell,face_number);
					  for(unsigned int q_point = 0;q_point<n_face_q_points;++q_point)
					  {
						  //Computation and storage of the shape functions on boundary face integration points
						  const FEFaceValues<dim> &fe_face_values = hp_fe_face_values.get_present_fe_values();
						  for(unsigned int i=0;i<dofs_per_cell;++i)
							  cell_rhs(i) += (fe_face_values.shape_value(i,q_point)*fe_face_values.JxW(q_point));

						  //Computation and addition of the terms in the RHS
						  cell->get_dof_indices(local_dof_indices);
						  for(unsigned int i=0;i<dofs_per_cell;++i)
							  system_rhs(local_dof_indices[i]) -= time_step*convection_coeff*cell_rhs(i)*((1-theta)*old_solution(i)-Tamb);
					  }

					  //Term to be added in the system matrix
					  hp_fe_values.reinit(cell);
					  const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

					  //Computation and storage of the value of the shape functions on boundary cell integration points
					  for(unsigned int q_point = 0;q_point < n_q_points; ++q_point)
						  for(unsigned int i=0;i<dofs_per_cell;++i)
							  for(unsigned int j=0;j<dofs_per_cell;++j)
								  cell_matrix(i,j) += (fe_values.shape_value(i,q_point)*fe_values.shape_value(j,q_point)*fe_values.JxW(q_point));

					  cell->get_dof_indices(local_dof_indices);
					  for(unsigned int i=0;i<dofs_per_cell;++i)
						  for(unsigned int j=0;j<dofs_per_cell;++j)
							  boundary_matrix.add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i,j));
				  }
			  }
		  }

		  //Computation and addition of the terms in the system matrix
		  system_matrix.add(theta*time_step*convection_coeff,boundary_matrix);
	  }
	  constraints.condense(system_matrix,system_rhs);
  }

  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
	SparseDirectUMFPACK		A_direct;

	//Initialization and LU factorization of A_direct
	A_direct.initialize(system_matrix);

	//Direct Resolution
	A_direct.vmult(solution,system_rhs);

	//Application of hanging nodes constraints
	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	constraints.close();
	constraints.distribute(solution);

	std::cout << "***Time step " << timestep_number << " at t=" << time
	<< std::endl;
	std::cout << std::endl
			<< "==========================================="
			<< std::endl
			<< "Number of active cells: " <<
			triangulation.n_active_cells()
			<< std::endl
			<< "Number of degrees of freedom: " <<
			dof_handler.n_dofs()
			<< std::endl
			<< std::endl;
  }

  template<int dim>
  bool HeatEquation<dim>::cell_is_in_metal_domain(const typename hp::DoFHandler<dim>::cell_iterator
		  &cell)
  {
	  bool in_metal=false;
	  unsigned int n = 0;
	  for (unsigned int v=0;v<GeometryInfo<dim>::vertices_per_cell;++v)
	  {
		  double limit = (1+floor(LaserSpeed*time))*layerThickness;
		  in_metal = (cell->vertex(n)[dim-1]) < std::max(part_height,limit);
		  if(in_metal==false)
			  n++;
	  }
	  return in_metal;
  }

  template<int dim>
  bool HeatEquation<dim>::cell_is_in_void_domain(const typename hp::DoFHandler<dim>::cell_iterator
		  &cell)
  {
	  bool in_void = false;
	  unsigned int n = 0;
	  for(unsigned int v=0;v<GeometryInfo<dim>::vertices_per_cell;++v)
	  {
		  double limit = (1+floor(LaserSpeed*time))*layerThickness;
		  in_void = cell->vertex(n)[dim-1] > std::max(part_height,limit);
		  if(in_void == false)
			  n++;
	  }
	  return in_void;
  }

  template<int dim>
  void HeatEquation<dim>::set_active_fe_indice()
  {

	  //Iteration over all the cells of the mesh
	  for(typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();cell != dof_handler.end();++cell)
	  {
		  //Lagrange Element if the cell is in metal domain
		  if(cell_is_in_metal_domain(cell))
			  cell->set_active_fe_index(0);
		  //Zero element if the cell is in void domain
		  else if (cell_is_in_void_domain(cell))
			  cell->set_active_fe_index(1);
		  //Throw an error if none of the above two cases is encountered
		  else
			  Assert(false,ExcNotImplemented());
	  }
  }

  template<int dim>
  void HeatEquation<dim>::part_height_measure()
  {
	  double max_height = part_height; //Maximal Height of vertice belonging to the metal domain in the previous function call

	  //Iteration over all the cells and storage of the maximal height of a vertex belonging to the metal domain
	  typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),endc = dof_handler.end();
	  for(;cell!=endc;++cell)
	  {
		  if(cell->active_fe_index() == 0)
		  {
			  double height_temp = 0;
			  for(unsigned int v=0;v<GeometryInfo<dim>::vertices_per_cell;++v)
			  {
				  height_temp = cell->vertex(v)[dim-1];
				  if(height_temp>max_height)
					  max_height = height_temp;
			  }
		  }
	  }
	  part_height = max_height;
  }


  template <int dim>
  void HeatEquation<dim>::output_results() const
  {
    DataOut<dim,hp::DoFHandler<dim>> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "T");

    data_out.build_patches();

    const std::string filename = "solution-"
                                 + Utilities::int_to_string(timestep_number, 3)
                                 + ".vtk";
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
  }

  template<int dim>
  void HeatEquation<dim>::store_old_vectors()
  {
	  map_old_solution.clear();
	  const MappingQ1<dim,dim> mapping;

	  typename hp::DoFHandler<dim>::active_cell_iterator cell1 = dof_handler.begin_active(),endc1 = dof_handler.end();
	  for(;cell1!=endc1;cell1++)
	  {
		  //Temporary variable to get the number of dof for the currently visited cell
		  const unsigned int dofs_per_cell = cell1->get_fe().dofs_per_cell;

		  std::vector<Point<dim>> support_points(dofs_per_cell);

		  if(dofs_per_cell != 0)//To skip the cell with FE = FE_Nothing as there is no support point there
		  {
			  //Get the coordinates of the support points on the unit cell
			  support_points = fe_collection[0].get_unit_support_points();

			  //Get the coordinates of the support points on the real cell
			  for(unsigned int i=0;i<dofs_per_cell;i++)
			  {
				  support_points[i] = mapping.transform_unit_to_real_cell(cell1,support_points[i]);
				  map_old_solution[support_points[i]] = VectorTools::point_value(dof_handler,old_solution,support_points[i]);
			  }
		  }
	  }
  }

  template <int dim>
  void HeatEquation<dim>::refine_mesh (const unsigned int min_grid_level,
                                       const unsigned int max_grid_level)
  {
	//Error estimation to set refine flags
	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

	//Computation of the vector of the KellyErrorEstimator on each cell

	KellyErrorEstimator<dim>::estimate(dof_handler, face_quadrature_collection, typename FunctionMap<dim>::type(), solution, estimated_error_per_cell);

	GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error_per_cell, 0.6, 0.4);

	//Clear the refine flag of the cell which are already at the maximal level of refinement
	if(triangulation.n_levels()>max_grid_level)
		for(typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(max_grid_level);cell != triangulation.end(); ++cell)
			cell->clear_refine_flag();

	// Considering the bug in deal.II library all coarsen flags are removed
	// Otherwise only the coarsening flag of the cells which are at the minimal level of refinement would be cleared

	for (typename Triangulation<dim>::active_cell_iterator
	cell = triangulation.begin_active();
	cell != triangulation.end(); ++cell)
			cell->clear_coarsen_flag ();

	//Computation of the new triangulation and DoFHandler
	triangulation.prepare_coarsening_and_refinement();
	SolutionTransfer<dim,Vector<double>,hp::DoFHandler<dim>> solution_trans(dof_handler);
	solution_trans.prepare_for_coarsening_and_refinement(solution);
	triangulation.execute_coarsening_and_refinement();
	dof_handler.distribute_dofs(fe_collection);

	//Solution interpolation on the new DoF Handler
	Vector<double> new_solution(dof_handler.n_dofs());
	solution_trans.interpolate(solution, new_solution);
	solution.reinit(dof_handler.n_dofs());
	solution = new_solution;

	old_solution = solution;

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	constraints.close();

	//Computation of the hanging node constraints
	constraints.distribute(solution);
  }

  template<int dim>
  void HeatEquation<dim>::transfer_old_vectors()
  {
	  //Creation of a solution of the same size of the number of dof of the new FE space
	  Vector<double> long_solution;
	  long_solution.reinit(dof_handler.n_dofs(),false);

	  const MappingQ1<dim,dim> mapping;

	  std::vector<types::global_dof_index> local_dof_indices;
	  //Iteration over all the cells of the updated dof_handler to fill the solution vector from the one from of the former one
	  unsigned int k=0;
	  typename hp::DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(),endc = dof_handler.end();
	  for(;cell!=endc;cell++)
	  {
		  //Vector of the points already stored
		  std::vector<Point<dim>> points_stored;

		  //Number of dofof the currently visited cell
		  const unsigned int dofs_per_cell = cell -> get_fe().dofs_per_cell;

		  //Vector of the support points of one cell
		  std::vector<Point<dim>> support_points(dofs_per_cell);

		  if(dofs_per_cell!=0)// To skip the cell with FE = FE_Nothing because they have not any support point
		  {
			  // Get the coordinates of the support points on the unit cell
			  support_points = fe_collection[0].get_unit_support_points();

			  // Vector of the degree of freedom indices of one cell
			  local_dof_indices.resize(dofs_per_cell);
			  cell->get_dof_indices(local_dof_indices);

			  //Get the coordinates of the support points on the real cell
			  for (unsigned int i=0;i<dofs_per_cell;i++)
			  {
				  support_points[i] =
				  mapping.transform_unit_to_real_cell(cell,
				  support_points[i]);
				  typename std::map< Point<dim>, double>::iterator
				  iter_old = map_old_solution.begin();
				  double solution_temp = 0;

				  // Variable to check if a point has already been stored
				  unsigned int is_point_stored = 0;

				  // Iteration in the old solution map
				  for(;iter_old!= map_old_solution.end();iter_old++)
				  {
					  // Test if the point visited corresponds to a point in the "old" dof_handler
					  if(support_points[i] == iter_old -> first )
						  //store the Point currently visited
						  points_stored.push_back(support_points[i]);
					  typename std::vector<Point<dim>>::iterator it_points = points_stored.begin();

					  //Test if the point has not been visited and stored yet
					  for(;it_points != points_stored.end();it_points++)
					  {
						  if(*it_points == iter_old->first)
							  is_point_stored++; //=1 if it is the first time the point is visited, >1 if it is not
					  }
				  }
				  //In case it has not been visited yet --> we store the solution
				  if(is_point_stored == 1)
				  {
					  solution_temp = map_old_solution.find(support_points[i])-> second;
					  //Write the solution at the right place inside the vector
					  long_solution[local_dof_indices[i]] = solution_temp;
					  k++;
				  }
			  }
		  }
	  }
	  solution.reinit(dof_handler.n_dofs());
	  old_solution = long_solution;
	  constraints.clear();
	  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	  constraints.close();
	  constraints.distribute(old_solution);
  }

  template <int dim>
  void HeatEquation<dim>::run()
  {
  //Initial Mesh Creation
  const unsigned int initial_global_refinement = 5; //minimum level of refinement
  const unsigned int n_adaptive_pre_refinement_steps = 7;// maximal level of refinement

  create_coarse_grid();
  triangulation.refine_global (initial_global_refinement);

  //Set the right FE Type for each cell
  set_active_fe_indice();

  //Initialize the matrices and RHS
  setup_system();
  old_solution.reinit(dof_handler.n_dofs());

  //Initial Condition

  //Instantiation
  InitialCondition<dim> initial_condition;

  //Projection of the function defined in the initial condition class on the limit of the domain
  VectorTools::project(dof_handler,constraints,quadrature_collection,initial_condition,old_solution,false,face_quadrature_collection);

  //Dirichlet Boundary Conditions

  //Instantiation
  //Setting up boundary_id for Dirichlet BC
  for(typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active();cell!=triangulation.end();cell++){
	  for(unsigned int f=0;f<GeometryInfo<dim>::faces_per_cell;f++){
		  if(std::abs(cell->face(f)->center()[1])<1e-12)
			  cell->face(f)->set_all_boundary_ids(1);
	  }
  }
  BoundaryValues<dim> boundary_value_function;
  boundary_value_function.set_time(time);

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,1,boundary_value_function,boundary_values);

  //Modifies old_solution vector, rhs vector and system matrix according to Dirichlet BCs

  MatrixTools::apply_boundary_values(boundary_values,system_matrix,old_solution,system_rhs);


  //Initialization of the solution vector taking into account the initial condition
  solution = old_solution;

  //Solution map filling
  store_old_vectors();

  std::cout << "***Time step " << timestep_number << " at t=" << time<< std::endl;
  std::cout << std::endl
  << "==========================================="
  << std::endl
  << "Number of active cells: " << triangulation.n_active_cells()
  << std::endl
  << "Number of degrees of freedom: " << dof_handler.n_dofs()
  << std::endl
  << std::endl;
  output_results();

  //Beginning of the time loop
  while(time <=1.2)
  {
	  time+=time_step;
	  ++timestep_number;

	  //Part height measurement considering the new layer
	  part_height_measure();

	  //Give the right FE type on each cell
	  set_active_fe_indice();

	  //Compute the mass and stiffness matrices on the new FE domain
	  setup_system();

	  //Transfer the solution(Temperature Field) to the new dofhandler
	  transfer_old_vectors();

	  //Compute the system matrix and RHS vector
	  assemble_system();

	  //Dirichlet boundary Conditions
	  //Setting up boundary_id for Dirichlet BC
	   for(typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active();cell!=triangulation.end();cell++){
	 	  for(unsigned int f=0;f<GeometryInfo<dim>::faces_per_cell;f++){
	 		  if(std::abs(cell->face(f)->center()[1])<1e-12)
	 			  cell->face(f)->set_all_boundary_ids(1);
	 	  }
	   }

	  BoundaryValues<dim> boundary_values_function;
	  boundary_values_function.set_time(time);

	  std::map<types::global_dof_index,double> boundary_values;
	  VectorTools::interpolate_boundary_values(dof_handler, 1, boundary_values_function, boundary_values);
	  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);


	  //Compute the vector of nodal temperatures with direct solver
	  solve_time_step();

	  //Adaptive Mesh Refinement
	  refine_mesh(initial_global_refinement,n_adaptive_pre_refinement_steps);
	  part_height_measure();

	  //Produce the output files
	  output_results();

	  //Fill the map matching each point to its temperature
	  store_old_vectors();
  }
  }
}


int main()
{
  try
    {
      using namespace dealii;
      using namespace AdditiveSimulation;

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
