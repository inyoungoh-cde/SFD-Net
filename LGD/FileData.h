// FileData.h
#ifndef FILEDATA_H
#define FILEDATA_H

#include <string>
#include <vector>
#include <Eigen/Dense>

struct FileData {
	std::string filename;
	std::vector<Eigen::Vector3d> positions;   // x, y, z coordinates
	std::vector<Eigen::VectorXd> extras;      // additional channel data (labels, etc.)
};

#endif // FILEDATA_H
