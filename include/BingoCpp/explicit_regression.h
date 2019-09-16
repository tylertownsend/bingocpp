#ifndef BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_

#include <vector>

#include <Eigen/Core>

#include "BingoCpp/equation.h"
#include "BingoCpp/fitness_function.h"
#include "BingoCpp/training_data.h"

namespace bingo {

struct ExplicitTrainingData : TrainingData {
  Eigen::ArrayXXd x;
  Eigen::ArrayXXd y;
  ExplicitTrainingData() : TrainingData() { }
  ExplicitTrainingData(Eigen::ArrayXXd input, Eigen::ArrayXXd output);
  ExplicitTrainingData* GetItem(int item);
  ExplicitTrainingData* GetItem(const std::vector<int>& items);
  int Size() { return x.rows(); }
};

class ExplicitRegression : public VectorBasedFunction {
 public:
  ExplicitRegression(ExplicitTrainingData* training_data) : 
      VectorBasedFunction(training_data) {}
  ~ExplicitRegression() {}
  Eigen::ArrayXXd EvaluateFitnessVector(const Equation& individual);
};
} // namespace bingo
#endif // BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_