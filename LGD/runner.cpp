#include "runner.h"
#include <atomic>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

int method::system(const std::string &outDir, int kset, const std::vector<FileData> &allFileData) {
	size_t totalFiles = allFileData.size();
	std::atomic<size_t> fileCounter(0);
	const int progressBarWidth = 50;  // width of the progress bar

	// Process the files in parallel (one file per iteration)
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < (int)totalFiles; ++j) {
		// Per-thread copies of the data for this file
		const FileData &data = allFileData[j];
		std::string outputName = outDir + data.filename;
		std::vector<Eigen::Vector3d> pos = data.positions;
		std::vector<Eigen::VectorXd> extra = data.extras;

		runmethod(pos, extra, outputName, kset);

		// Atomic counter keeps the progress count race-free
		size_t count = ++fileCounter;
		double progress = double(count) / totalFiles;
		int posB = int(progressBarWidth * progress);

		// Console output is serialized so the progress bar stays readable
#pragma omp critical
		{
			std::cout << "\r[";
			for (int i = 0; i < progressBarWidth; ++i) {
				if (i < posB)       std::cout << "=";
				else if (i == posB) std::cout << ">";
				else                std::cout << " ";
			}
			std::cout << "] "
				<< std::setw(3) << int(progress * 100)
				<< "% (" << count << "/" << totalFiles << ")"
				<< std::flush;
		}
	}
	std::cout << std::endl;
	return 0;
}

int method::runmethod(std::vector<Eigen::Vector3d> &pos, std::vector<Eigen::VectorXd> &extra, std::string &out, int kNeighbors) {
	try {
		estimator ne;
		// The kNN search returns the query point itself as one of the results,
		// so search with k + 1 to keep k actual neighbors.
		ne.set_K(kNeighbors + 1);
		ne.SetXYZ(pos);

		ne.m_proposed();

		// Save: x y z + geometric features (3 channels) + label (if present)
		ne.outgeneratedFeatures(out, extra);
	}

	catch (std::exception& e) {
		std::cerr << "Unhandled Exception reached the top of main: "
			<< e.what() << ", application will now exit " << std::endl;
		return 1;
	}
	return 0;
}
