/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
#include <math.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

 // Defining M here:
#define NUM_PARTICLES 100

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_psi(theta, std[2]);

  for (int i = 0; i < NUM_PARTICLES; ++i) {
    double sample_x, sample_y, sample_psi;
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_psi(gen);
    //p.id = i; // <- this isnt used anywhere
    p.weight = 1;
    this->particles.push_back(p);
  }
  this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  for (Particle & p : this->particles) {
    double newtheta = p.theta + yaw_rate*delta_t;
    p.x += velocity * (sin(newtheta) - sin(p.theta)) / yaw_rate;
    p.y += velocity * (cos(p.theta) - cos(newtheta)) / yaw_rate;
    p.theta = newtheta;

    normal_distribution<double> dist_x(p.x, std[0]);
    normal_distribution<double> dist_y(p.y, std[1]);
    normal_distribution<double> dist_psi(p.theta, std[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_psi(gen);
  }

}

#define SQUARE(x) (x)*(x)

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks_map, std::vector<LandmarkObs>& observations_map) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  // iterate through observations
  for (auto & obs : observations_map) {
    obs.id = -1;
    double min_dist_sq = INT64_MAX;

    // check distance to every landmark, map to nearest
    for (auto &pred : landmarks_map) {
      double dist_sq = SQUARE(obs.y - pred.y) + SQUARE(obs.x - pred.x);
      if (min_dist_sq > dist_sq) {
        min_dist_sq = dist_sq;
        obs.id = pred.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
  std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html
  //
  //
  // // Note: above advice is not correct.
      // =================================
      // It should be  + y.sin(t) only for a left-handed coordinate system like screen coordinates
      // in this case it must be -ve.
      //
  //

  // Precompute some terms
  double two_sigma_x_sq = 2 * SQUARE(std_landmark[0]);
  double two_sigma_y_sq = 2 * SQUARE(std_landmark[1]);
  double one_over_two_pi_sigma_sq = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

  this->weights.clear();
  for (int i = 0; i < NUM_PARTICLES; ++i) {
    Particle p = this->particles[i];

    // Convert meaurement to map space based on particles position
    double cost = cos(p.theta);
    double sint = sin(p.theta);
    vector<LandmarkObs> observations_tx;
    for (int j = 0; j < observations.size(); j++) {
      // Translate to map space
      LandmarkObs obs_m;
      obs_m.x = (observations[j].x * cost - observations[j].y * sint) + p.x;
      obs_m.y = (observations[j].x * sint + observations[j].y * cost) + p.y;
      observations_tx.push_back(obs_m);
    }

    // Select landmarks that are in range
    vector<LandmarkObs> mapLandmarks;
    for (auto &landmark : map_landmarks.landmark_list)
    {
      if (SQUARE(p.x - landmark.x_f) + SQUARE(p.y - landmark.y_f) <= SQUARE(sensor_range)) {
        LandmarkObs l;
        l.id = landmark.id_i;
        l.x = landmark.x_f;
        l.y = landmark.y_f;
        mapLandmarks.push_back(l);
      }
    }

    // Find landmarks that best fit measurement
    dataAssociation(mapLandmarks, observations_tx);

    // Compute weight
    p.weight = 1;
    for (auto& obs : observations_tx) {
      if (obs.id != -1 && p.weight > 0) {
        //if (map_landmarks.landmark_list[obs.id - 1].id_i != obs.id)
         // throw new exception("unexpected map item");
        double delta_x = obs.x - map_landmarks.landmark_list[obs.id - 1].x_f;
        double delta_y = obs.y - map_landmarks.landmark_list[obs.id - 1].y_f;

        double prob = one_over_two_pi_sigma_sq * exp(-((SQUARE(delta_x) / two_sigma_x_sq) + (SQUARE(delta_y) / two_sigma_y_sq)));
        p.weight *= prob;
      }
    }
    this->weights.push_back(p.weight);
  }

}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  discrete_distribution<int> dist(this->weights.begin(), this->weights.end());
  vector<Particle> new_particles;
  // pick particles with replacement
  for (int i = 0; i < NUM_PARTICLES; i++) {
    new_particles.push_back(this->particles[dist(gen)]);
  }
  this->particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);
  for (int i = 0; i < num_particles; ++i) {
    dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
  }
  dataFile.close();
}
