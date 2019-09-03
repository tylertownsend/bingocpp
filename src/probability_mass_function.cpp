#include <stdexcept>
#include <iostream>
#include <numeric>
#include <random>

#include <BingoCpp/probability_mass_function.h>

namespace bingo {
namespace {

template<typename T>
void set_default_weights(Eigen::VectorXd& weights, const std::vector<T>& items) {
  if (items.size() > 0) {
    weights = Eigen::VectorXd::Constant(items.size(), 1.0/items.size());
  }
}

template<typename T>
void weights_and_items_have_equal_size(const std::vector<T>& items,
                                       const Eigen::VectorXd& weights) {
  if (weights.size() != items.size()) {
    std::cout << "Initialization of ProbabilityMassFunction with "
              << "items and weights of different dimensions\n";
    throw std::invalid_argument("Invalid argument.");
  }
}

void check_valid_weights(const Eigen::VectorXd& weights,
                         const Eigen::VectorXd& normalize_weights) {
  if ((normalize_weights.size() > 0) &&
      ((1 - normalize_weights.sum() > 1e-6) ||
      (weights.minCoeff() < 0.0))) {
        std::cout << "Invalid weights encountered "
                  << "in ProbabilityMassFunction" << std::endl;
    throw std::invalid_argument("Invalid argument.");
  }
}

void normalize_weights(const Eigen::VectorXd& weights,
                       double& total_weight,
                       Eigen::VectorXd& normalized_weights) {
  double PERCENTAGE = 100;
  total_weight = weights.sum();
  normalized_weights = weights / total_weight;
  check_valid_weights(normalized_weights, weights);
  normalized_weights *= PERCENTAGE;
}

double get_mean_current_weight(const double& total_weight,
                               const Eigen::VectorXd& normalized_weights) {
  if (normalized_weights.size() == 0) {
    return 1.0;
  }
  return total_weight / normalized_weights.size();
}
} // namespace 

template<typename T>
ProbabilityMassFunction<T>::ProbabilityMassFunction() {
  items_ = std::vector<T>();
  weights_ = Eigen::VectorXd(0);
  total_weight_ = 0;
  normalized_weights_ = Eigen::VectorXd(0);
}

template<typename T>
ProbabilityMassFunction<T>::ProbabilityMassFunction(
    std::vector<T> items,
    Eigen::VectorXd weights) {
  items_ = items;
  weights_ = weights_;
  total_weight_ = weights_.sum();
  normalized_weights_ = Eigen::VectorXd(0);
}

template<typename T>
ProbabilityMassFunction<T>::ProbabilityMassFunction(
    const ProbabilityMassFunction& pmf) {
  items_ = pmf.items_;
  weights_ = pmf.weights_;
  total_weight_ = pmf.total_weight_;
  normalized_weights_ = pmf.normalized_weights_;
}

template<typename T>
void ProbabilityMassFunction<T>::addItem(T new_item) {
  double new_weight = get_mean_current_weight(total_weight_,
                                              normalized_weights_);
  addItem(new_item, new_weight);
}

template<typename T>
void ProbabilityMassFunction<T>::addItem(T new_item, double new_weight) {
  items_.push_back(new_item);
  Eigen::VectorXd weights(normalized_weights_.size() + 1);
  weights << total_weight_ * normalized_weights_, new_weight;
  weights_ = weights;
  normalize_weights(weights_, total_weight_, normalized_weights_);
}

template<typename T>
T ProbabilityMassFunction<T>::drawSample() {
  std::random_device rd;
  std::mt19937 engine(rd());
  double* the_data = normalized_weights_.data();
  std::discrete_distribution<int> dist(the_data);
  return items_[dist(engine)];
}
} // namespace bingo