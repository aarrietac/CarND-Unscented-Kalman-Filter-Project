#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

double Tools::NormRad(double *angle) {
  while (*angle >  M_PI) *angle -= 2*M_PI;
  while (*angle < -M_PI) *angle += 2*M_PI;
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  // initializes RMSE vector
  VectorXd rmse = VectorXd::Zero(4);

  // check viability of inputs dimension
  int n_est = estimations.size();
  int n_gtr = ground_truth.size();
  if (n_est != n_gtr || n_est == 0){
    cout << "Invalid estimations or ground_truth dimension" << endl;
    return rmse;
  }

  // compute RMSE
  for (int i = 0; i < n_est; i++){
    VectorXd err_ = estimations[i] - ground_truth[i];
    err_ = err_.array() * err_.array();
    rmse += err_;
  }
  rmse /= n_est;
  rmse = rmse.array().sqrt();

  return rmse;
}
