#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // number of states
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // for a normal person in a bike, 1.5 m/s^2 seems like a quite good
  // aproximation for the longitudinal acceleration
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // tested with 2*pi/3, pi/4 and pi/3 (better results and realistic yaw acc.)
  std_yawdd_ = M_PI/3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // set RADAR space dimension
  n_radar_ = 3;

  // initialize RADAR measurement noise covariance matrix
  R_radar_ = MatrixXd::Zero(n_radar_, n_radar_);
  R_radar_(0, 0) = std_radr_*std_radr_;
  R_radar_(1, 1) = std_radphi_*std_radphi_;
  R_radar_(2, 2) = std_radrd_*std_radrd_;

  // set LASER space dimension
  n_laser_ = 2;

  // initialize LASER measurement noise covariance matrix
  R_laser_ = MatrixXd::Zero(n_laser_, n_laser_);
  R_laser_(0, 0) = std_laspx_*std_laspx_;
  R_laser_(1, 1) = std_laspy_*std_laspy_;

  // number of process noise variables
  n_noise_ = 2;

  // set process noise covariance matrix
  Q_ = MatrixXd::Zero(n_noise_, n_noise_);
  Q_(0, 0) = std_a_*std_a_;
  Q_(1, 1) = std_yawdd_*std_yawdd_;

  // Initialized flag
  is_initialized_ = false;

  // Augmented dimension
  n_aug_ = 7;

  // set total number of sigma points
  n_sig_ = 2*n_aug_ + 1;

  // create matrix of sigma points
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // create Weights vector
  weights_ = VectorXd::Zero(n_sig_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_){
    // first measurement
    cout << "UKF: " << endl;

    time_us_ = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);

      // from polar to cartesian
      x_ << rho*cos(phi), rho*sin(phi), 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
      x_ << meas_package.raw_measurements_(0),  // px
            meas_package.raw_measurements_(1),  // py
            0, 0, 0;  // no velocity and yaw measurement from LASER
    }
    is_initialized_ = true;
    return;
  }

  // compute delta_t between t(k+1) to t(k)
  double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    UpdateLidar(meas_package);
  }
  return;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // import tools for angle normalization
  Tools tools;

  //create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  // create augmented covariance matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  // compute augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q_;

  // calculate square root of P
  MatrixXd A = MatrixXd(n_aug_, n_aug_);
  A = P_aug.llt().matrixL();

  // compute augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++){
      Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_)*A.col(i);
      Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*A.col(i);
  }

  // vector states, time derivative and stochastic part
  VectorXd x = VectorXd(n_x_);
  VectorXd dx = VectorXd(n_x_);
  VectorXd noise_x = VectorXd(n_x_);

  double delta_t2 = delta_t*delta_t;

  for (int i = 0; i < n_sig_; i++){
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double dyaw = Xsig_aug(4, i);
    double m_a = Xsig_aug(5, i);
    double m_ddyaw = Xsig_aug(6, i);

    // current state vector
    x << px, py, v, yaw, dyaw;

    // time derivative of the current state vector
    double yawt = yaw + dyaw*delta_t;
    double vdyaw = v/dyaw;
    if (fabs(dyaw) > 0.001){
      dx << vdyaw*(sin(yawt) - sin(yaw)),
            vdyaw*(-cos(yawt) + cos(yaw)),
            0,
            dyaw*delta_t,
            0;
    }else{
      dx << v*cos(yaw)*delta_t, v*sin(yaw)*delta_t, 0, 0, 0;
    }

    // stochastic part of the system
    noise_x << 0.5*delta_t2*cos(yaw)*m_a,
               0.5*delta_t2*sin(yaw)*m_a,
               delta_t*m_a,
               0.5*delta_t2*m_ddyaw,
               delta_t*m_ddyaw;

    Xsig_pred_.col(i) = x + dx + noise_x;
  }

  //set weights
  weights_.fill(0.5/(lambda_ + n_aug_));
  weights_(0) = lambda_/(lambda_ + n_aug_);

  //predict state mean (vectorial form)
  x_ = Xsig_pred_ * weights_;

  //predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    tools.NormRad(&(x_diff(3)));

    P_ = P_ + weights_(i)*x_diff*x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  int n_z = n_laser_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);

  // update state vector and covariance state matrix
  UpdatePX(meas_package, Zsig, n_z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  int n_z = n_radar_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, n_sig_);

  //transform sigma points into measurement space
  for (int i = 0; i < 2*n_aug_ + 1; i++){
    double px = Xsig_pred_.col(i)(0);
    double py = Xsig_pred_.col(i)(1);
    double v = Xsig_pred_.col(i)(2);
    double yaw = Xsig_pred_.col(i)(3);

    // compute rho
    double rho = sqrt(px*px + py*py);
    if (rho < 0.001){
      cout << "Error: division by zero" << endl;
      break;
    }
    Zsig.col(i)(0) = rho;

    // compute phi angle
    if (fabs(px) < 0.001 && fabs(py) > 0.001){
      Zsig.col(i)(1) = atan2(py, 0);
    } else{
      Zsig.col(i)(1) = atan2(py, px);
    }

    // compute rho_dot
    Zsig.col(i)(2) = (px*cos(yaw) + py*sin(yaw))*v/rho;
  }

  // update state vector and covariance state matrix
  UpdatePX(meas_package, Zsig, n_z);
}

/**
 * Compute matrices and vectors to update the state and the state covariance
 * @param {MeasurementPackage} meas_package
 * @param {MatrixXd} Zsig
 * @param {int} n_z
 */
void UKF::UpdatePX(MeasurementPackage meas_package, MatrixXd Zsig, int n_z) {

  // import tools for angle normalization
  Tools tools;

  // mean predicted state vector from sigma points
  VectorXd z_pred = Zsig * weights_;

  // calculate measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < n_sig_; i++){
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // normalize yaw angle
      tools.NormRad(&(z_diff(1)));

      S = S + weights_(i)*z_diff*z_diff.transpose();
  }

  // compute the measurement noise covariance matrix
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    R = R_radar_;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    R = R_laser_;
  }
  S = S + R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  //calculate cross correlation matrix
  VectorXd x_diff = VectorXd::Zero(n_x_);
  VectorXd z_diff = VectorXd::Zero(n_z);

  for (int i=0; i < 2*n_aug_ + 1; i++){
      x_diff = Xsig_pred_.col(i) - x_;
      z_diff = Zsig.col(i) - z_pred;

      // normalize yaw angle
      tools.NormRad(&(x_diff(3)));
      tools.NormRad(&(z_diff(1)));

      Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
  }

  //calculate Kalman gain K
  MatrixXd Sinv = S.inverse();
  MatrixXd K = Tc*Sinv;

  // take measurement data
  VectorXd z = meas_package.raw_measurements_;

  // normalize yaw angle if necessary
  z_diff = z - z_pred;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    tools.NormRad(&(z_diff(1)));
  }

  // update state mean and covariance matrix
  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*K.transpose();

  // compute NIS
  double NIS = z_diff.transpose()*Sinv*z_diff;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    NIS_radar_ = NIS;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    NIS_laser_ = NIS;
  }
}
