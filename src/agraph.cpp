#include <limits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <BingoCpp/agraph.h>
#include <BingoCpp/backend.h>

const double kNaN = std::numeric_limits<double>::quiet_NaN();

const PrintMap kStackPrintMap {
  {2, "({}) + ({})"},
  {3, "({}) - ({})"},
  {4, "({}) * ({})"},
  {5, "({}) / ({}) "},
  {6, "sin ({})"},
  {7, "cos ({})"},
  {8, "exp ({})"},
  {9, "log ({})"},
  {10, "({}) ^ ({})"},
  {11, "abs ({})"},
  {12, "sqrt ({})"},
};

const PrintMap kLatexPrintMap {
  {2, "{} + {}"},
  {3, "{} - ({})"},
  {4, "({})({})"},
  {5, "\\frac{ {} }{ {} }"},
  {6, "sin{ {} }"},
  {7, "cos{ {} }"},
  {8, "exp{ {} }"},
  {9, "log{ {} }"},
  {10, "({})^{ ({}) }"},
  {11, "|{}|"},
  {12, "\\sqrt{ {} }"},
};

const PrintMap kConsolePrintMap {
  {2, "{} + {}"},
  {3, "{} - ({})"},
  {4, "({})({})"},
  {5, "({})/({})"},
  {6, "sin({})"},
  {7, "cos({})"},
  {8, "exp({})"},
  {9, "log({})"},
  {10, "({})^({})"},
  {11, "|{}|"},
  {12, "sqrt({})"},
};

std::string print_string_with_args(const std::string& string,
                                   std::string& arg1,
                                   std::string& arg2) {
  std::stringstream stream;
  bool first_found = false;
  for (auto character = string.begin(); character != string.end(); character++) {
    if (*character == '{' && *(character+1) == '}') {
      stream << ((!first_found) ? arg1 : arg2);
      character++;
      first_found = true;
    } else {
      stream << *character;
    }
  }
  return stream.str();
}

namespace bingo {
namespace {
std::string get_stack_element_string(const AGraph& individual,
                                     int command_index ) {
  Eigen::ArrayX3i command_array = individual.getCommandArray();
  int node = command_array(command_index, 0);
  int param1 = command_array(command_index, 1);
  int param2 = command_array(command_index, 2);

  std::string temp_string = "("+ std::to_string(command_index) +") <= ";
  if (node == 0) {
    temp_string += "X_" + std::to_string(param1);
  } else if (node == 1) {
    if (param1 == -1 ||
        param1 >= individual.getNumberLocalOptimizationParams()) {
      temp_string += "C";
    } else {
      Eigen::VectorXd parameter = individual.getLocalOptimizationParams();
      temp_string += "C_" + std::to_string(param1) + " = " + 
                     std::to_string(parameter[param1]);
    }
  } else {
    std::string param1_str = std::to_string(param1);
    std::string param2_str = std::to_string(param2);
    temp_string += print_string_with_args(kStackPrintMap.at(node),
                                         param1_str,
                                         param2_str);
  }
  temp_string += '\n';
  return temp_string;
}

std::string get_stack_string(const AGraph& individual, const bool is_short=false) {
  std::vector<bool> commands;
  if (is_short) {
    commands = individual.getUtilizedCommands();
  } else {
    commands.resize(individual.getCommandArray().rows());
    std::fill(commands.begin(), commands.end(), 1);
  }
  std::string temp_string;
  for (size_t i = 0; i < commands.size(); i++) {
    if (commands[i]) {
      temp_string += get_stack_element_string(individual, i);
    }
  }
  return temp_string;
}

std::string get_formatted_element_string(const AGraph& individual,
                                         const int command_index,
                                         std::vector<std::string> string_list,
                                         const PrintMap& format_map) {
  Eigen::ArrayX3i command_array = individual.getCommandArray();
  int node = command_array(command_index, 0);
  int param1 = command_array(command_index, 1);
  int param2 = command_array(command_index, 2);

  std::string temp_string;
  if (node == 0) {
    temp_string = "X_" + std::to_string(param1);
  } else if (node == 1) {
    if (param1 == -1 ||
        param1 >= individual.getNumberLocalOptimizationParams()) {
      temp_string = "?";
    } else {
      Eigen::VectorXd parameter = individual.getLocalOptimizationParams();
      temp_string = std::to_string(parameter[param1]);
    }
  } else {
    temp_string = print_string_with_args(format_map.at(node),
                                         string_list[param1],
                                         string_list[param2]);
  }
  return temp_string;
}

std::string get_formatted_string_using(const AGraph& indvidual,
                                       const PrintMap& format_map) {
  std::vector<bool> utilized_rows = indvidual.getUtilizedCommands();
  std::vector<std::string> string_list;
  for (size_t i = 0; i < utilized_rows.size(); i++) {
    std::string temp_string;
    if (utilized_rows[i]) {
      temp_string = get_formatted_element_string(indvidual, i, string_list,
                                                 format_map);
      string_list.push_back(temp_string);
    }
  }
  return string_list.back();
}
} // namespace

AGraph::AGraph() {
  command_array_ = Eigen::ArrayX3i(0, 3);
  constants_ = Eigen::VectorXd(0);
  fitness_ = 1e9;
  fit_set_ = false;
  genetic_age_ = 0;
}

AGraph::AGraph(const AGraph& agraph) {
  command_array_ = agraph.getCommandArray();
  constants_ = agraph.getLocalOptimizationParams();
  fitness_ = agraph.getFitness();
  fit_set_ = agraph.isFitnessSet();
  genetic_age_ = agraph.getGeneticAge();
}

AGraph AGraph::copy() {
  AGraph return_val = AGraph(*this);
  return return_val;
}

Eigen::ArrayX3i AGraph::getCommandArray() const {
  return command_array_;
}

void AGraph::setCommandArray(Eigen::ArrayX3i command_array) {
  command_array_ = command_array;
  fitness_ = 1e9;
  fit_set_ = false;
}

void AGraph::notifyCommandArrayModificiation() {
  fitness_ = 1e9;
  fit_set_ = false;
}

double AGraph::getFitness() const {
  return fitness_;
}

void AGraph::setFitness(double fitness) {
  fitness_ = fitness;
}

bool AGraph::isFitnessSet() const {
  return fitness_;
}

void AGraph::setGeneticAge(const int age) {
  genetic_age_ = age;
}

int AGraph::getGeneticAge() const {
  return genetic_age_;
}

std::vector<bool> AGraph::getUtilizedCommands() const {
  return backend::getUtilizedCommands(command_array_);
}

bool AGraph::needsLocalOptimization() {
  std::vector<bool> commands = this->getUtilizedCommands();
  for (size_t row = 0; row < commands.size(); row ++) {
    if (commands[row] && (command_array_(row, 0) == 1)) {
      if (command_array_(row, 1) == -1 ||
          command_array_(row, 1) >= constants_.size()) {
        return true;
      }
    }
  }
  return false;
}

// TODO: fix with new update
int AGraph::getNumberLocalOptimizationParams() const {
  return constants_.size(); 
}

void AGraph::setLocalOptimizationParams(Eigen::VectorXd params) {
  constants_ = params;
}

Eigen::VectorXd AGraph::getLocalOptimizationParams() const {
  return constants_;
}

Eigen::ArrayXXd AGraph::evaluateEquationAt(Eigen::ArrayXXd& x) {
  Eigen::ArrayXXd f_of_x; 
  try {
    f_of_x = backend::evaluate(this->command_array_,
                               x,
                               this->constants_);
    return f_of_x;
  } catch (const std::underflow_error& ue) {
    return Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
  } catch (const std::overflow_error& oe) {
    return Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
  } 
}

EvalAndDerivative AGraph::evaluateEquationWithXGradientAt(Eigen::ArrayXXd& x) {
  EvalAndDerivative df_dx;
  try {
    df_dx = backend::evaluateWithDerivative(this->command_array_,
                                            x,
                                            this->constants_,
                                            true);
    return df_dx;
  } catch (const std::underflow_error& ue) {
    Eigen::ArrayXXd nan_array = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  } catch (const std::overflow_error& oe) {
    Eigen::ArrayXXd nan_array = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  }
}

EvalAndDerivative AGraph::evaluateEquationWithLocalOptGradientAt(
    Eigen::ArrayXXd& x) {
  EvalAndDerivative df_dc;
  try {
    df_dc = backend::evaluateWithDerivative(this->command_array_,
                                            x,
                                            this->constants_,
                                            false);
    return df_dc;
  } catch (const std::underflow_error& ue) {
    Eigen::ArrayXXd nan_array = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  } catch (const std::overflow_error& oe) {
    Eigen::ArrayXXd nan_array = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  }
}

std::ostream& operator<<(std::ostream& strm, const AGraph& graph) {
  return strm << graph.getConsoleString();
}

std::string AGraph::getLatexString() const {
  return get_formatted_string_using(*this, kLatexPrintMap);
}

std::string AGraph::getConsoleString() const {
  return get_formatted_string_using(*this, kConsolePrintMap);
}

std::string AGraph::getStackString() const {
  std::stringstream print_str; 
  print_str << "---full stack---\n"
            << get_stack_string(*this)
            << "---small stack---\n"
            << get_stack_string(*this, true); 
  return print_str.str();
  
}

int AGraph::getComplexity() const {
  std::vector<bool> commands = getUtilizedCommands();
  return std::count_if (commands.begin(), commands.end(), [](bool i) {
    return i;
  });
}

const bool AGraph::kIsArity2Map[13] = {
  false,
  false,
  true,
  true,
  true,
  true,
  false,
  false,
  false,
  false,
  true,
  false,
  false
};

const bool AGraph::kIsTerminalMap[13] = {
  true,
  true,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false
};
}