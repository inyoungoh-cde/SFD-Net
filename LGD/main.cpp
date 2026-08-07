#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>  // for std::setw
#include <string>
#include <vector>
#include <cctype>
#include <ctime>

#include "platform.h"
#include "runner.h"
#include "FileData.h"

// ---------------------------------------------------------------------------
// Usage:  LGD [input_dir] [output_dir] [k]
//   input_dir  : folder with the input .txt files (default: "data")
//   output_dir : folder for the results          (default: "output_YYMMDD")
//   k          : k-NN neighborhood size          (default: 40)
// Relative paths are resolved against the project root (the folder holding
// LGD.sln / CMakeLists.txt).
// Arguments are order-flexible: a purely numeric argument is taken as k,
// the first non-numeric one as input_dir and the second as output_dir.
// ---------------------------------------------------------------------------

static const std::string DEFAULT_INPUT_DIR = "data";
static const int DEFAULT_K = 40;

method mrun;

void run(const std::string &inDir, const std::string &outDir, int kset);

std::vector<std::string> dirReader(const std::string &name);
int countChannelsInFirstTxt(const std::vector<std::string>& files, const std::string &dirPath);
std::vector<FileData> processTxtFiles(const std::vector<std::string>& files, const std::string &dirPath, int channels);

// Today's date as YYMMDD (e.g. 260807)
static std::string dateStamp() {
	std::tm tmv = localTimeNow();
	char buf[16];
	std::strftime(buf, sizeof(buf), "%y%m%d", &tmv);
	return std::string(buf);
}

// Markers that exist only in the source root. LGD.sln is deliberately not one
// of them: CMake's Visual Studio generator writes its own LGD.sln into the
// build directory, which would make the search stop one level too early.
static bool isProjectRoot(const fs::path &p) {
	return fs::exists(p / "CMakeLists.txt") || fs::exists(p / "third_party_includes");
}

// Project root = the folder holding the sources. Walking up from the
// executable makes the same binary work from x64\Release (MSBuild), from
// build/Release (CMake) or next to the sources. A deployed exe finds no
// marker and uses its own folder instead.
static fs::path findBaseDir() {
	const std::string exePath = executablePath();
	fs::path exeDir = exePath.empty() ? fs::current_path() : fs::path(exePath).parent_path();

	for (fs::path p = exeDir; !p.empty(); p = p.parent_path()) {
		try {
			if (isProjectRoot(p)) return p;
		}
		catch (const std::exception&) {
			// Unreadable directory: keep walking up
		}
		if (p == p.root_path()) break;
	}
	// Fallback for a deployed exe: the folder next to the executable
	return exeDir;
}

static bool isNumber(const std::string &s) {
	if (s.empty()) return false;
	for (char c : s) {
		if (!std::isdigit(static_cast<unsigned char>(c))) return false;
	}
	return true;
}

static std::string trim(const std::string &s) {
	std::string t = s;
	// Strip a UTF-8 BOM (can appear when input is piped/redirected from a file)
	if (t.size() >= 3 && (unsigned char)t[0] == 0xEF && (unsigned char)t[1] == 0xBB && (unsigned char)t[2] == 0xBF) {
		t.erase(0, 3);
	}
	size_t b = t.find_first_not_of(" \t\r\n");
	if (b == std::string::npos) return "";
	size_t e = t.find_last_not_of(" \t\r\n");
	return t.substr(b, e - b + 1);
}

// Ask for a value on the console; an empty answer keeps the default
static std::string promptString(const std::string &label, const std::string &defVal) {
	std::cout << label << " [" << defVal << "]: ";
	std::string line;
	if (!std::getline(std::cin, line)) return defVal;
	line = trim(line);
	return line.empty() ? defVal : line;
}

static int promptInt(const std::string &label, int defVal) {
	while (true) {
		std::cout << label << " [" << defVal << "]: ";
		std::string line;
		if (!std::getline(std::cin, line)) return defVal;
		line = trim(line);
		if (line.empty()) return defVal;
		if (isNumber(line)) return std::stoi(line);
		std::cout << "Please enter a positive number." << std::endl;
	}
}

// Keep the console window open when launched by double-click
static void pauseBeforeExit() {
	std::cout << std::endl << "Press Enter to exit...";
	std::string dummy;
	std::getline(std::cin, dummy);
}

// Create the output directory (including missing parent folders)
static void makeDirectory(const fs::path &path) {
	if (fs::exists(path)) {
		std::cout << "Directory already exists: " << path.string() << std::endl;
	}
	else if (fs::create_directories(path)) {
		std::cout << "Created directory: " << path.string() << std::endl;
	}
	else {
		std::cerr << "Failed to create directory: " << path.string() << std::endl;
	}
}

static void printUsage() {
	std::cout <<
		"Usage: LGD [input_dir] [output_dir] [k]\n"
		"  input_dir  : folder with input .txt files (default: data)\n"
		"  output_dir : folder for results           (default: output_" << dateStamp() << ")\n"
		"  k          : k-NN neighborhood size       (default: " << DEFAULT_K << ")\n"
		"Relative paths are resolved against the project root.\n";
}

int main(int argc, char *argv[]) {
	std::string inputArg = DEFAULT_INPUT_DIR;
	std::string outputArg = "output_" + dateStamp();
	int kset = DEFAULT_K;

	// No command-line arguments (double-click / IDE run):
	// ask interactively, Enter keeps the default shown in brackets.
	const bool interactive = (argc <= 1);
	if (interactive) {
		std::cout << "LGD - geometric feature extractor" << std::endl;
		std::cout << "(press Enter to accept the default shown in [brackets])" << std::endl << std::endl;
		inputArg = promptString("Input folder ", inputArg);
		outputArg = promptString("Output folder", outputArg);
		kset = promptInt("k-NN size    ", kset);
		std::cout << std::endl;
	}
	else {
		// Flexible argument parsing: numbers -> k, strings -> input then output
		int stringArgCount = 0;
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-h" || arg == "--help" || arg == "/?") {
				printUsage();
				return 0;
			}
			if (isNumber(arg)) {
				kset = std::stoi(arg);
			}
			else if (stringArgCount == 0) {
				inputArg = arg;
				++stringArgCount;
			}
			else if (stringArgCount == 1) {
				outputArg = arg;
				++stringArgCount;
			}
			else {
				std::cerr << "Ignoring extra argument: " << arg << std::endl;
			}
		}
	}

	if (kset <= 0) {
		std::cerr << "Invalid k value: " << kset << std::endl;
		if (interactive) pauseBeforeExit();
		return 1;
	}

	const fs::path base = findBaseDir();

	// Absolute arguments are used as-is; relative ones are resolved against base
	fs::path inDir = fs::path(inputArg).is_absolute() ? fs::path(inputArg) : base / inputArg;
	fs::path outDir = fs::path(outputArg).is_absolute() ? fs::path(outputArg) : base / outputArg;

	std::cout << "Base directory   : " << base.string() << std::endl;
	std::cout << "Input directory  : " << inDir.string() << std::endl;
	std::cout << "Output directory : " << outDir.string() << std::endl;
	std::cout << "k-NN size        : " << kset << std::endl << std::endl;

	if (!fs::exists(inDir) || !fs::is_directory(inDir)) {
		std::cerr << "Input directory not found: " << inDir.string() << std::endl << std::endl;
		printUsage();
		if (interactive) pauseBeforeExit();
		return 1;
	}

	makeDirectory(outDir);

	// Ensure a trailing separator so filenames can be appended directly.
	// '/' is accepted by both Windows and Linux file APIs.
	std::string outDirStr = outDir.string();
	if (!outDirStr.empty() && outDirStr.back() != '/' && outDirStr.back() != '\\') {
		outDirStr += '/';
	}

	run(inDir.string(), outDirStr, kset);

	if (interactive) pauseBeforeExit();
	return 0;
}

void run(const std::string &inDir, const std::string &outDir, int kset) {
	std::vector<std::string> dir_load = dirReader(inDir);

	// Count the number of channels from the first .txt file in the directory
	int Channels = countChannelsInFirstTxt(dir_load, inDir);
	if (Channels < 3) {
		std::cerr << "Input files must have at least 3 channels (x y z). Found: " << Channels << std::endl;
		return;
	}
	std::cout << "loaded files: " << dir_load.size() << " with " << Channels << " channels." << std::endl;

	// Load every file into memory (one FileData entry per file)
	std::vector<FileData> allFileData = processTxtFiles(dir_load, inDir, Channels);

	std::cout << "Running the method!!" << std::endl;
	mrun.system(outDir, kset, allFileData);
}

std::vector<std::string> dirReader(const std::string &name) {
	std::vector<std::string> vecArray;
	for (auto & entry : fs::directory_iterator(name)) {
		vecArray.push_back(entry.path().filename().string());
	}
	return vecArray;
}

// Count the number of channels (columns) in the first line of the first .txt file
int countChannelsInFirstTxt(const std::vector<std::string>& files, const std::string &dirPath) {
	int returnChannels = 0;
	std::string txtFilename;

	// Find the first file with a ".txt" extension in the file list
	for (const auto &file : files) {
		if (file.size() >= 4 && file.substr(file.size() - 4) == ".txt") {
			txtFilename = file;
			break;
		}
	}

	if (txtFilename.empty()) {
		std::cout << "No .txt file found in the directory." << std::endl;
		return 0;
	}

	std::string fullPath = dirPath + "/" + txtFilename;
	std::ifstream inFile(fullPath);
	if (!inFile) {
		std::cerr << "Failed to open file: " << fullPath << std::endl;
		return 0;
	}

	std::string line;
	if (std::getline(inFile, line)) {
		std::istringstream iss(line);
		double value;
		int channelCount = 0;
		// Count the whitespace-separated numbers one by one
		while (iss >> value) {
			++channelCount;
		}
		returnChannels = channelCount;
	}
	else {
		std::cout << "File is empty: " << fullPath << std::endl;
	}
	return returnChannels;
}

std::vector<FileData> processTxtFiles(const std::vector<std::string>& files, const std::string &dirPath, int channels) {
	std::vector<FileData> fileDataVec;
	size_t totalFiles = files.size();
	size_t fileCounter = 0;
	const int progressBarWidth = 50;  // width of the progress bar

	// Process only the .txt files in the file list
	for (const auto &file : files) {
		if (file.size() >= 4 && file.substr(file.size() - 4) == ".txt") {
			std::string fullPath = dirPath + "/" + file;
			std::ifstream inFile(fullPath);
			if (!inFile) {
				std::cerr << "Failed to open file: " << fullPath << std::endl;
				continue;
			}
			FileData fdata;
			fdata.filename = file;
			std::string line;
			while (std::getline(inFile, line)) {
				if (line.empty()) continue;

				std::istringstream iss(line);
				std::vector<double> values;
				double num;
				while (iss >> num) {
					values.push_back(num);
				}
				if (values.empty()) continue;
				if (values.size() != static_cast<size_t>(channels)) {
					std::cerr << "Warning: " << fullPath << " has a line with " << values.size()
						<< " values (expected " << channels << "). Skipping this line." << std::endl;
					continue;
				}
				fdata.positions.push_back(Eigen::Vector3d(values[0], values[1], values[2]));
				if (channels > 3) {
					int extraCount = channels - 3;
					Eigen::VectorXd extra(extraCount);
					for (int i = 0; i < extraCount; ++i) {
						extra(i) = values[i + 3];
					}
					fdata.extras.push_back(extra);
				}
			} // while getline

			fileDataVec.push_back(fdata);
			++fileCounter;

			// Update the progress bar on a single line
			double progress = static_cast<double>(fileCounter) / totalFiles;
			int pos = static_cast<int>(progressBarWidth * progress);
			std::cout << "\r[";
			for (int i = 0; i < progressBarWidth; ++i) {
				if (i < pos)
					std::cout << "=";
				else if (i == pos)
					std::cout << ">";
				else
					std::cout << " ";
			}
			std::cout << "] " << std::setw(3) << int(progress * 100) << "% ("
				<< fileCounter << "/" << totalFiles << ")" << std::flush;
		}
	}
	std::cout << std::endl;  // line break after completion
	return fileDataVec;
}
