/*!
 * \file utils.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains utility functions for doing and testing
 * sybolic regression problems in the bingo package
 */

#include <iostream>
#include <vector>
#include <numeric>

#include "BingoCpp/utils.h"

#include <iostream>

namespace bingo {

const int kPartialWindowSize = 7;
const int kPartialEdgeSize = 3;
const int kDerivativeOrder = 1;

void set_break_points(const Eigen::ArrayXXd &x,
                      std::vector<int> *break_points) {
  for (int i = 0; i < x.rows(); ++i) {
    if (std::isnan(x(i))) {
      break_points->push_back(i);
    }
  }
  break_points->push_back(x.rows());
}

void update_return_values (int start,
                           const Eigen::ArrayXXd &x_segment,
                           const Eigen::ArrayXXd &time_deriv,
                           Eigen::ArrayXXd *x_return,
                           Eigen::ArrayXXd *time_deriv_return) {
  if (start ==  0) {
    x_return->resize(x_segment.rows(), x_segment.cols());
    *x_return << x_segment;
    time_deriv_return->resize(time_deriv.rows(), time_deriv.cols());
    *time_deriv_return << time_deriv;
    std::cout << "time_driv in update values start = " << start << "\n";
    std::cout << *time_deriv_return << std::endl;
  } else {
    Eigen::ArrayXXd x_temp = *x_return;
    x_return->resize(x_return->rows() + x_segment.rows(), x_return->cols());
    *x_return << x_temp, x_segment;
    Eigen::ArrayXXd deriv_temp = *time_deriv_return;
    time_deriv_return->resize(time_deriv_return->rows() + time_deriv.rows(),
                              time_deriv_return->cols());
    *time_deriv_return << deriv_temp, time_deriv;
    std::cout << "time_driv in update values start = " << start << "\n";
    std::cout << *time_deriv_return << std::endl;
  }
}

InputAndDeriviative CalculatePartials(const Eigen::ArrayXXd &x) {
  std::vector<int> break_points;
  set_break_points(x, &break_points);

  int start = 0;
  int return_value_rows = (x.rows() - kPartialEdgeSize) * break_points.size();
  Eigen::ArrayXXd x_return(return_value_rows, x.cols());
  Eigen::ArrayXXd time_deriv_return(x_return.rows(), x_return.cols());
  std::cout << "break_points" << std::endl;
  for (auto i = 0; i < break_points.size(); i++) { std::cout << break_points[i] << std::endl;}
  for (std::vector<int>::iterator break_point = break_points.begin();
      break_point != break_points.end();
      break_point ++) {
    Eigen::ArrayXXd x_segment = x.block(start, 0, 
                                        *break_point - start,
                                        x.cols());
    Eigen::ArrayXXd time_deriv(x_segment.rows(), x_segment.cols());

    for (int col = 0; col < x_segment.cols(); col ++) {
      auto return_value = SavitzkyGolay(x_segment.col(col),
                                          kPartialWindowSize,
                                          kPartialEdgeSize,
                                          kDerivativeOrder);
      std::cout << "return value from SG" << return_value << std::endl;
      time_deriv.col(col) = return_value;
    }

    time_deriv = time_deriv.block(kPartialEdgeSize, 0, 
                                  time_deriv.rows() - kPartialWindowSize,
                                  time_deriv.cols());
    x_segment = x_segment.block(kPartialEdgeSize, 0,
                                x_segment.rows() - kPartialWindowSize,
                                x_segment.cols());
    
    std::cout << "break point " << *break_point << std::endl;
    std::cout << "x_segment\n" << x_segment << std::endl;
    std::cout << "time deriv\n" << time_deriv << std::endl;
    update_return_values(start, x_segment, time_deriv, 
                         &x_return, &time_deriv_return);
    start = *break_point + 1;
  }
  return std::make_pair(x_return, time_deriv_return);
}

double GramPoly(double eval_point,
                double num_points,
                double polynomial_order,
                double derivative_order) {

  double result = 0;
  if (polynomial_order > 0) {
    result = (4. * polynomial_order - 2.) / 
             (polynomial_order * (2. * num_points - polynomial_order + 1.)) *
             (eval_point *
                 GramPoly(eval_point, num_points, 
                          polynomial_order - 1.,
                          derivative_order)
             +
             derivative_order *
                GramPoly(eval_point, num_points,
                         polynomial_order - 1.,
                         derivative_order - 1.))
             -
             ((polynomial_order - 1.) * (2. * num_points + polynomial_order)) /
             (polynomial_order * (2. * num_points - polynomial_order + 1.)) *
             GramPoly(eval_point, num_points,
                      polynomial_order - 2,
                      derivative_order);
  } else if (polynomial_order == 0 && derivative_order == 0) {
    result = 1.;
  } else {
    result = 0.;
  }
  return result;
}

double GenFact(double a, double b) {
  int fact = 1;
  for (int i = a - b + 1; i < a + 1; ++i) {
    fact *= i;
  }
  return fact;
}

double GramWeight(double eval_point_start,
                  double eval_point_end,
                  double num_points,
                  double ploynomial_order,
                  double derivative_order) {
  double weight = 0;

  for (int i = 0; i < ploynomial_order + 1; ++i) {
    weight += (2. * i + 1.) * GenFact(2. * num_points, i) /
              GenFact(2. * num_points + i + 1, i + 1) *
              GramPoly(eval_point_start, num_points, i, 0) *
              GramPoly(eval_point_end, num_points, i, derivative_order);
  }

  return weight;
}

Eigen::ArrayXXd convolution(const Eigen::ArrayXXd &data_points,
                            int half_filter_size,
                            const Eigen::ArrayXXd &weights) {
  int data_points_center = 0;
  int w_ind = 0;
  int data_points_len = data_points.rows();
  Eigen::ArrayXd convolution(data_points_len, 1);

  for (int i = 0; i < data_points_len; ++i) {
    if (i < half_filter_size) {
      data_points_center = half_filter_size;
      w_ind = i;

    } else if (data_points_len - i <= half_filter_size) {
      data_points_center = data_points_len - half_filter_size - 1;
      w_ind = 2 * half_filter_size + 1 - (data_points_len - i);

    } else {
      data_points_center = i;
      w_ind = half_filter_size;
    }
    convolution(i) = 0;
    // std::cout << "convolution" << std::endl;
    for (int j = half_filter_size * -1; j < half_filter_size + 1; ++j) {
      // std::cout << "j = " << j << std::endl;
      convolution(i) += data_points(data_points_center + j)
                       * weights(j + half_filter_size, w_ind);
      // std::cout << convolution << std::endl;
    }
  }
  return convolution;
}

Eigen::ArrayXXd SavitzkyGolay(Eigen::ArrayXXd y,
                              int window_size,
                              int polynomial_order,
                              int derivative_order) {
  int m = (window_size - 1) / 2;
  Eigen::ArrayXXd weights(2 * m + 1, 2 * m + 1);
  for (int i = m * -1; i < m + 1; ++i) {
    for (int j = m * -1; j < m + 1; ++j) {
      weights(i + m, j + m) = 
        GramWeight(i, j, m, polynomial_order, derivative_order);
    }
  }
  return convolution(y, m, weights);
}
} // namespace bingo