#ifndef algorithm_ours_HEADER
#define algorithm_ours_HEADER

#include <vector>
#include <string>
#include <fstream>
#include <cmath>   // std::isnan
#include <ctime>   // time
#include <cstdlib> // rand, srand

#include <Eigen/Dense>
#include <nanoflann.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
typedef __int64 pointIdxType;
#else
typedef long int pointIdxType;
#endif

class estimator {

protected:

	Eigen::MatrixX3d pts;          /*!< Point cloud (one point per row) */
	Eigen::MatrixXd est_featureXd; /*!< Estimated per-point features (output) */

	int neighborhood_size;         /*!< k-NN neighborhood size */

public:

	typedef nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixX3d > kd_tree; //a row is a point

	estimator() {
		neighborhood_size = 200;
	}

	const int& get_K() const { return neighborhood_size; }
	void set_K(int K) {
		neighborhood_size = K;
	}

	void SetXYZ(std::vector<Eigen::Vector3d> &pos) {
		pts.resize(pos.size(), 3);
		for (int i = 0; i < pts.rows(); i++) {
			pts.row(i) << pos[i].x(), pos[i].y(), pos[i].z();
		}
	}

	// Returns 0.0 when the value is NaN
	double safeValue(double val) {
		return std::isnan(val) ? 0.0 : val;
	}

	// Writes one line per point: x y z + estimated features + label (if any).
	// The label is taken from the last column of the original file.
	void outgeneratedFeatures(const std::string& output, std::vector<Eigen::VectorXd> &extra) {
		std::ofstream ofs(output.c_str());
		const bool hasLabel = ((int)extra.size() == (int)pts.rows());
		for (int i = 0; i < pts.rows(); i++) {
			ofs << pts(i, 0) << " ";
			ofs << pts(i, 1) << " ";
			ofs << pts(i, 2);
			for (int j = 0; j < est_featureXd.cols(); j++) {
				ofs << " " << safeValue(est_featureXd(i, j));
			}
			if (hasLabel) ofs << " " << extra[i][extra[i].size() - 1];
			if (i != pts.rows() - 1) ofs << std::endl;
		}
		ofs.close();
	}

	void m_proposed() {
		// Random permutation of the points (balances the dynamic OpenMP schedule)
		srand((unsigned int)time(NULL));
		std::vector<int> permutation(pts.rows());
		for (int i = 0; i < pts.rows(); i++) {
			permutation[i] = i;
		}
		for (int i = 0; i < pts.rows(); i++) {
			int j = rand() % pts.rows();
			int temp = permutation[i];
			permutation[i] = permutation[j];
			permutation[j] = temp;
		}

		// kd tree creation
		kd_tree tree(3, pts, 10);
		tree.index->buildIndex();

		Eigen::MatrixX3d s0nls(pts.rows(), 3), s1nls(pts.rows(), 3), s2nls(pts.rows(), 3);
		// Define the three neighborhood sizes
		int scale0_neighborhood_size = neighborhood_size / 2; // Scale 0: Half of the original neighborhood size
		int scale1_neighborhood_size = neighborhood_size;     // Scale 1: Original neighborhood size
		int scale2_neighborhood_size = neighborhood_size * 2; // Scale 2: Twice the original neighborhood size

#ifdef _OPENMP
		omp_set_num_threads(omp_get_max_threads()); // Set the number of threads to the maximum available
#endif
		// #1. PCA-based initial normal estimation
#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
		for (int per = 0; per < (int)pts.rows(); per++) {
			// Index of the point
			int n = permutation[per];

			// Getting the list of neighbors for both scales (using the larger scale as the loop range)
			std::vector<pointIdxType> pointIdxSearch_s2; // Use the larger scale for the knn search
			std::vector<double> pointSquaredDistance_s2;

			const Eigen::Vector3d& pt_query = pts.row(n);
			pointIdxSearch_s2.resize(scale2_neighborhood_size);
			pointSquaredDistance_s2.resize(scale2_neighborhood_size);
			tree.index->knnSearch(&pt_query[0], scale2_neighborhood_size, &pointIdxSearch_s2[0], &pointSquaredDistance_s2[0]);

			// Initialize matrices to store neighbors' points for each scale
			Eigen::MatrixXd pointsXd_s0(scale0_neighborhood_size, 3); // Smaller scale points
			Eigen::MatrixXd pointsXd_s1(scale1_neighborhood_size, 3); // Middle scale points
			Eigen::MatrixXd pointsXd_s2(scale2_neighborhood_size, 3); // Larger scale points

																	  // Loop over the neighbors (up to the larger scale)
			for (int pt = 0; pt < (int)pointIdxSearch_s2.size(); pt++) {
				pointsXd_s2.row(pt) = pts.row(pointIdxSearch_s2[pt]); // Store points for the larger scale

																	  // For the smaller scales, only store points within the smaller range
				if (pt < scale1_neighborhood_size) {
					pointsXd_s1.row(pt) = pts.row(pointIdxSearch_s2[pt]); // Use the same index but store fewer points
				}
				if (pt < scale0_neighborhood_size) {
					pointsXd_s0.row(pt) = pts.row(pointIdxSearch_s2[pt]); // Use the same index but store fewer points
				}
			}
			// STEP 1: PCA-based normal estimation for Scale 0 (smallest scale)
			Eigen::Vector3d centroid_s0 = pointsXd_s0.colwise().mean(); // Calculate centroid for scale 0

			Eigen::MatrixXd centered_s0 = pointsXd_s0.rowwise() - centroid_s0.transpose(); // Subtract centroid to move points to origin
			Eigen::Matrix3d cov_s0 = centered_s0.transpose() * centered_s0; // Compute covariance matrix for scale 0

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_s0(cov_s0);
			Eigen::Vector3d normal_s0 = es_s0.eigenvectors().col(0); // Smallest eigenvalue's eigenvector is the normal


			Eigen::Vector3d centroid_s1 = pointsXd_s1.colwise().mean(); // Calculate centroid for scale 1
			Eigen::MatrixXd centered_s1 = pointsXd_s1.rowwise() - centroid_s1.transpose(); // Subtract centroid to move points to origin
			Eigen::Matrix3d cov_s1 = centered_s1.transpose() * centered_s1; // Compute covariance matrix for scale 1

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_s1(cov_s1);
			Eigen::Vector3d normal_s1 = es_s1.eigenvectors().col(0); // Smallest eigenvalue's eigenvector is the normal


																	 // STEP 2: PCA-based normal estimation for Scale 2 (larger scale)
			Eigen::Vector3d centroid_s2 = pointsXd_s2.colwise().mean(); // Calculate centroid for scale 2
			Eigen::MatrixXd centered_s2 = pointsXd_s2.rowwise() - centroid_s2.transpose(); // Subtract centroid to move points to origin
			Eigen::Matrix3d cov_s2 = centered_s2.transpose() * centered_s2; // Compute covariance matrix for scale 2

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_s2(cov_s2);
			Eigen::Vector3d normal_s2 = es_s2.eigenvectors().col(0); // Smallest eigenvalue's eigenvector is the normal

			normal_s0 /= normal_s0.norm(); // Normalize the normal
			normal_s1 /= normal_s1.norm(); // Normalize the normal
			normal_s2 /= normal_s2.norm(); // Normalize the normal

			s0nls.row(n) << normal_s0(0), normal_s0(1), normal_s0(2); // Store the normal in s0nls

			s1nls.row(n) << normal_s1(0), normal_s1(1), normal_s1(2); // Store the normal in s1nls

			s2nls.row(n) << normal_s2(0), normal_s2(1), normal_s2(2); // Store the normal in s2nls
		}

		est_featureXd.resize(pts.rows(), 3);

		// #2. CPS (Candidate Point Selection) for both scales
#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
		for (int per = 0; per < (int)pts.rows(); per++) {
			// Index of the point
			int n = permutation[per];

			// Getting the list of neighbors (using the larger scale as the loop range)
			std::vector<pointIdxType> pointIdxSearch_s2;
			std::vector<double> pointSquaredDistance_s2;

			const Eigen::Vector3d& pt_query = pts.row(n);
			pointIdxSearch_s2.resize(scale2_neighborhood_size); // Use the defined scale2_neighborhood_size
			pointSquaredDistance_s2.resize(scale2_neighborhood_size);
			tree.index->knnSearch(&pt_query[0], scale2_neighborhood_size, &pointIdxSearch_s2[0], &pointSquaredDistance_s2[0]);

			// Initialize matrices to store neighbors' normal differences for both scales
			Eigen::MatrixXd normalDiff_s0(scale0_neighborhood_size, 3); // Use the defined scale0_neighborhood_size
			Eigen::MatrixXd normalDiff_s1(scale1_neighborhood_size, 3); // Use the defined scale1_neighborhood_size
			Eigen::MatrixXd normalDiff_s2(scale2_neighborhood_size, 3); // Use the defined scale2_neighborhood_size
			double SS_s0 = 0.0, SS_s1 = 0.0, SS_s2 = 0.0; // Sum of weights for both scales

			for (int j = 0; j < scale2_neighborhood_size; ++j) { // Use scale2_neighborhood_size explicitly
																 // Compute the normal difference for the larger scale (s2)
				Eigen::Vector3d n_c_0, n_c_1, n_c_2,
					n_k_0, n_k_1, n_k_2;
				n_k_2 << s2nls(pointIdxSearch_s2[j], 0), s2nls(pointIdxSearch_s2[j], 1), s2nls(pointIdxSearch_s2[j], 2);
				n_c_2 << s2nls(n, 0), s2nls(n, 1), s2nls(n, 2);

				double vdiff_s2 = (n_k_2 - n_c_2).norm();
				double weight_s2 = (vdiff_s2 != 0.0) ? vdiff_s2 * vdiff_s2 : 1.0;
				normalDiff_s2.row(j) = sqrt(weight_s2) * (n_k_2 - n_c_2).transpose();
				SS_s2 += weight_s2;

				// For the middle scale, process only within the middle range
				if (j < scale1_neighborhood_size) { // Use scale1_neighborhood_size explicitly
					n_k_1 << s1nls(pointIdxSearch_s2[j], 0), s1nls(pointIdxSearch_s2[j], 1), s1nls(pointIdxSearch_s2[j], 2);
					n_c_1 << s1nls(n, 0), s1nls(n, 1), s1nls(n, 2);

					double vdiff_s1 = (n_k_1 - n_c_1).norm();
					double weight_s1 = (vdiff_s1 != 0.0) ? vdiff_s1 * vdiff_s1 : 1.0;
					normalDiff_s1.row(j) = sqrt(weight_s1) * (n_k_1 - n_c_1).transpose();
					SS_s1 += weight_s1;
				}

				// For the smallest scale, process only within the smallest range
				if (j < scale0_neighborhood_size) { // Use scale0_neighborhood_size explicitly
					n_k_0 << s0nls(pointIdxSearch_s2[j], 0), s0nls(pointIdxSearch_s2[j], 1), s0nls(pointIdxSearch_s2[j], 2);
					n_c_0 << s0nls(n, 0), s0nls(n, 1), s0nls(n, 2);

					double vdiff_s0 = (n_k_0 - n_c_0).norm();
					double weight_s0 = (vdiff_s0 != 0.0) ? vdiff_s0 * vdiff_s0 : 1.0;
					normalDiff_s0.row(j) = sqrt(weight_s0) * (n_k_0 - n_c_0).transpose();
					SS_s0 += weight_s0;
				}
			}

			// Compute covariance matrix and eigenvalues for the smallest scale (s0)
			Eigen::MatrixXd cov_s0 = (normalDiff_s0.transpose() * normalDiff_s0) / SS_s0;
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_s0(cov_s0);
			double wi3_s0 = 0.0;
			if (eig_s0.eigenvalues().sum() != 0.0) {
				wi3_s0 = 1 - (eig_s0.eigenvalues()(2) / eig_s0.eigenvalues().sum());
			}

			Eigen::MatrixXd cov_s1 = (normalDiff_s1.transpose() * normalDiff_s1) / SS_s1;
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_s1(cov_s1);
			double wi3_s1 = 0.0;
			if (eig_s1.eigenvalues().sum() != 0.0) {
				wi3_s1 = 1 - (eig_s1.eigenvalues()(2) / eig_s1.eigenvalues().sum());
			}

			// Compute covariance matrix and eigenvalues for the larger scale (s2)
			Eigen::MatrixXd cov_s2 = (normalDiff_s2.transpose() * normalDiff_s2) / SS_s2;
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_s2(cov_s2);
			double wi3_s2 = 0.0;
			if (eig_s2.eigenvalues().sum() != 0.0) {
				wi3_s2 = 1 - (eig_s2.eigenvalues()(2) / eig_s2.eigenvalues().sum());
			}

			// Each thread writes to a distinct row n, so no synchronization is needed
			est_featureXd.row(n) << wi3_s0, wi3_s1, wi3_s2;
		}
	}
};

#endif
