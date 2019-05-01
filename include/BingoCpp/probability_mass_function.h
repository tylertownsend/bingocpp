#ifndef INCLUDE_BINGOCPP_PROBABILITY_MASS_FUNCTION_H_
#define INCLUDE_BINGOCPP_PROBABILITY_MASS_FUNCTION_H_

#include <vector>

#include <Eigen/Dense>

namespace bingo {

template<typename T>
class ProbabilityMassFunction {
 private:
  std::vector<T> items_;
  Eigen::VectorXd weights_;
  double total_weight_;
  Eigen::VectorXd normalized_weights_;

 public:
  ProbabilityMassFunction();
  ProbabilityMassFunction(std::vector<T> items, Eigen::VectorXd weights);
  ProbabilityMassFunction(const ProbabilityMassFunction& pmf);
  void addItem(T new_item);
  void addItem(T new_item, double new_weight);
  T drawSample();
};
}
#endif // INCLUDE_BINGOCPP_PROBABILITY_MASS_FUNCTION_H_