/*
 * Point_Comparison_Operator.hpp
 *
 *  Created on: 03-May-2021
 *      Author: pushkar
 */

#ifndef POINT_COMPARISON_OPERATOR_HPP_
#define POINT_COMPARISON_OPERATOR_HPP_

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor_base.h>
#include <deal.II/base/point.h>
#include <cmath>

DEAL_II_NAMESPACE_OPEN

/**
* Comparison operator overloading. @relates Point
*/

template <int dim, typename Number>
inline
bool operator < (const Point<dim,Number> &p1, const Point<dim,Number> &p2)
{
	unsigned int j=0;
	for(unsigned int i=0;i<dim;i++)
	{
		if(p1[i]!=p2[i])
			return p1[j]<p2[j];
		else
			j++;
	}
	if(j==dim)
		return false;
}

template <int dim, typename Number>
inline
bool operator > (const Point<dim,Number> &p1, const Point<dim,Number> &p2)
{
	unsigned int j=0;
	for(unsigned int i=0;i<dim;i++)
	{
		if(p1[i]!=p2[i])
			return p1[j]>p2[j];
		else
			j++;
	}
	if(j==dim)
		return false;
}

template<int dim,typename Number>
inline
bool operator == (const Point<dim,Number> &p1,const Point<dim,Number> &p2)
{
	unsigned int j=0;
	for(unsigned int i=0;i<dim;i++)
	{
		if(p1[i]==p2[i])
			j++;
	}
	if(j==dim)
		return true;
	else
		return false;
}

DEAL_II_NAMESPACE_CLOSE
#endif /* POINT_COMPARISON_OPERATOR_HPP_ */
