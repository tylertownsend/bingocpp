#include "performance_benchmarks.h"

int main() {
  do_benchmarking();
  return 0;
}

void do_benchmarking() {
  BenchMarkTestData benchmark_test_data =  BenchMarkTestData();
  load_benchmark_data(benchmark_test_data);
  run_benchmarks(benchmark_test_data);
}

void load_benchmark_data(BenchMarkTestData &benchmark_test_data) {
  std::vector<AGraphValues> indv_list= std::vector<AGraphValues>();
  load_agraph_indvidual_data(indv_list);
  Eigen::ArrayXXd x_vals = load_agraph_x_vals();
  benchmark_test_data = BenchMarkTestData(indv_list, x_vals);
}

void load_agraph_indvidual_data(std::vector<AGraphValues> &indv_list) {
  std::ifstream stack_filestream;
  std::ifstream const_filestream;
  stack_filestream.open(STACK_FILE);
  const_filestream.open(CONST_FILE);

  std::string stack_file_line;
  std::string const_file_line;
  while ((stack_filestream >> stack_file_line) && 
      (const_filestream >> const_file_line)) {
    AGraphValues curr_indv = AGraphValues();
    set_indv_stack(curr_indv, stack_file_line);
    set_indv_constants(curr_indv, const_file_line);
    indv_list.push_back(curr_indv);
  }
  stack_filestream.close();
  const_filestream.close();
}

void set_indv_constants(AGraphValues &indv, std::string &const_string) {
  std::stringstream string_stream(const_string);
  std::string num_constants;
  std::getline(string_stream, num_constants, ',');
  Eigen::VectorXd curr_const = Eigen::VectorXd(std::stoi(num_constants));

  std::string curr_val;
  for (int i=0; std::getline(string_stream, curr_val, ','); i++) {
    curr_const(i) = std::stod(curr_val);
  }
  indv.constants = curr_const;
}

void set_indv_stack(AGraphValues &indv, std::string &stack_string) {
  std::stringstream string_stream(stack_string);
  Eigen::ArrayX3i curr_stack = Eigen::ArrayX3i(STACK_SIZE, 3);

  std::string curr_op;
  for (int i=0; std::getline(string_stream, curr_op, ','); i++) {
    curr_stack(i/3, i%3) = std::stoi(curr_op);
  }
  indv.command_array = curr_stack;
}


Eigen::ArrayXXd load_agraph_x_vals() {
  std::ifstream filename;
  filename.open(X_FILE);

  Eigen::ArrayXXd x_vals = Eigen::ArrayXXd(NUM_DATA_POINTS, INPUT_DIM);
  std::string curr_x_row;
  for (int row = 0; filename >> curr_x_row; row++) {
    std::stringstream string_stream(curr_x_row);
    std::string curr_x;
    for (int col = 0; std::getline(string_stream, curr_x, ','); col++) {
      x_vals(row, col) = std::stod(curr_x);
    }
  }
  filename.close();
  return x_vals;
}

void run_benchmarks(BenchMarkTestData &benchmark_test_data) {
  Eigen::ArrayXd evaluate_times = time_benchmark(benchmark_evaluate, benchmark_test_data);
  Eigen::ArrayXd x_derivative_times = time_benchmark(benchmark_evaluate_w_x_derivative, benchmark_test_data);
  Eigen::ArrayXd c_derivative_times = time_benchmark(benchmark_evaluate_w_c_derivative, benchmark_test_data);
  print_header();
  print_results(evaluate_times);
  print_results(x_derivative_times);
  print_results(c_derivative_times);
}

Eigen::ArrayXd time_benchmark(
  void (*benchmark)(std::vector<AGraphValues>&, Eigen::ArrayXXd&), 
  BenchMarkTestData &test_data, int number, int repeat) {
  Eigen::ArrayXd times = Eigen::ArrayXd(repeat);
  for (int run=0; run<repeat; run++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<number; i++) {
      benchmark(test_data.indv_list, test_data.x_vals);	
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::chrono::seconds> time_span = (stop - start);
    times(run) = time_span.count();
  }
  return times; 
}

void benchmark_evaluate(std::vector<AGraphValues> &indv_list,
                        Eigen::ArrayXXd &x_vals) {
  std::vector<AGraphValues>::iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    SimplifyAndEvaluate(indv->command_array,
                        x_vals,
                        indv->constants);
  } 
}

void benchmark_evaluate_w_x_derivative(std::vector<AGraphValues> &indv_list,
                                       Eigen::ArrayXXd &x_vals) {
  std::vector<AGraphValues>::iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    SimplifyAndEvaluateWithDerivative(indv->command_array,
                                      x_vals,
                                      indv->constants,
                                      true);
  }
}

void benchmark_evaluate_w_c_derivative(std::vector<AGraphValues> &indv_list,
                                       Eigen::ArrayXXd &x_vals) {
  std::vector<AGraphValues>::iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    SimplifyAndEvaluateWithDerivative(indv->command_array,
                                      x_vals,
                                      indv->constants,
                                      false);
  }
}

void print_header() {
  const std::string top_tacks = std::string(23, '-');
  const std::string title = ":::: PERFORMANCE BENCHMARKS ::::";
  const std::string full_title = top_tacks + title + top_tacks;
  const std::string bottom = std::string (78, '-');
}

void print_results(Eigen::ArrayXd &run_times) {
  double std_dev = standard_deviation(run_times);
  double average = run_times.mean();
  double max = run_times.maxCoeff;
  double min = run_times.minCoeff;
  std::cout<<std_dev<<std::endl;
  std::cout<<average<<std::endl;
  std::cout<<max<<std::endl;
  std::cout<<min<<std::endl;
  
}

double standard_deviation(Eigen::ArrayXd &vec) {
  return std::sqrt((vec - vec.mean()).square().sum()/(vec.size()-1));
}
