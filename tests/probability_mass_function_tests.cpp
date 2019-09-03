#include <stdexcept>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>

#include <BingoCpp/probability_mass_function.h>

using namespace bingo;

class PMFTest : public ::testing::Test {
 public:
  ProbabilityMassFunction<bool> constant_pmf;
  ProbabilityMassFunction<double> sample_pmf;
  ProbabilityMassFunction<int> empty_pmf;
  virtual void SetUp() {
    constant_pmf = init_constant_pmf();
    sample_pmf = init_sample_pmf();
    empty_pmf = ProbabilityMassFunction<int>();
  }
  virtual void TearDown() {}

 private:
  ProbabilityMassFunction<bool> init_constant_pmf() {
    Eigen::VectorXd weights(2);
    weights << 1, 0;
    std::vector<bool> items = {true, false};
    ProbabilityMassFunction pmf(items, weights);
    return pmf;
  }

  ProbabilityMassFunction<double> init_sample_pmf() {
    Eigen::VectorXd weights(4);  
    weights << 4.0, 3.0, 2.0, 1.0;
    std::vector<double> items = {1.0, 2.0, 3.0, 4.0};
    ProbabilityMassFunction pmf(items, weights);
    return pmf;
  }
};

TEST(PMFTest, raises_exception_for_uneven_init) {
  try {
    Eigen::VectorXd weights(1);
    weights << 1;
    std::vector<int> items = {1, 2 3};
    ProbabilityMassFunction<int> pmf(items, weights);
  } catch (const std::invalid_argument& err) {
    EXPECT_THAT(err.what(), std::string("Invalid argument"));
  } catch (...) {
    FAIL() << "Expected std::invalid_argument";
  }
}