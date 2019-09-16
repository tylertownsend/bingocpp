#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <BingoCpp/explicit_regression.h>

#include "test_fixtures.h"

using namespace bingo;

namespace {

class TestExplicitRegression : public testing::Test {
 public:
  Eigen::ArrayXXd x_;
  Eigen::ArrayXXd y_;
  ExplicitTrainingData* training_data_;
  testutils::SumEquation sum_equation_;

  void SetUp() {
    training_data_ = init_sample_training_data();
    sum_equation_ = testutils::init_sum_equation();
  }

  void TearDown() {
    delete training_data_;
  }

 private:
  ExplicitTrainingData* init_sample_training_data() {
    const int num_points = 50;
    const int num_data_per_feature = 10;
    const int num_feature = 50 / num_data_per_feature;
    Eigen::ArrayXXd x = Eigen::ArrayXd::LinSpaced(num_points, 0, 0.98);
    x = x.reshaped(num_feature, num_data_per_feature);
    x.transposeInPlace();
    Eigen::Array<double, 10, 1> y = Eigen::ArrayXd::LinSpaced(10, 0.2, 4.7);
    return new ExplicitTrainingData(x, y);
  }
};

TEST_F(TestExplicitRegression, HELLOWORLD) {
  ExplicitRegression regressor(training_data_);
  Eigen::ArrayXXd fitness = regressor.EvaluateFitnessVector(sum_equation_);
  Eigen::ArrayXXd zero = Eigen::ArrayXXd::Zero(10, 1);
  ASSERT_TRUE(testutils::almost_equal(fitness, zero, 1e-1));
}
} // namespace 