#ifndef runner_HEADER
#define runner_HEADER

#include "platform.h"   // defines the fs namespace alias
#include "algorithms.h"
#include "FileData.h"
#include <iomanip>  // for std::setw
#include <string>
#include <vector>

class method {
public:
	int system(const std::string &outDir, int kset, const std::vector<FileData> &allFileData);
	int runmethod(std::vector<Eigen::Vector3d> &pos, std::vector<Eigen::VectorXd> &extra, std::string &out, int kNeighbors);
};

#endif
