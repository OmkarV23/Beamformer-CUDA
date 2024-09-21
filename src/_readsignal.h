#ifndef READ_SIGNAL_H
#define READ_SIGNAL_H

#include <iostream>      // For standard I/O operations (std::cout, std::cerr)
#include <fstream>       // For file stream operations (std::ifstream)
#include <sstream>       // For string stream operations (std::stringstream)
#include <string>        // For std::string
#include <vector>        // For std::vector
#include <boost/filesystem.hpp>
#include "auxilary.h"

namespace fs = boost::filesystem;

std::vector<std::vector<std::string>> readCSV(const std::string& fileName);

std::vector<std::vector<double>> readAllCSVFiles(const std::string& directoryPath);

void saveToCSV(const std::vector<double>& values, const std::string& filename);

#endif // READ_SIGNAL_H
