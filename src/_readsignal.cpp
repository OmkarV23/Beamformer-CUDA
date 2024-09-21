#include "_readsignal.h"

std::vector<std::vector<std::string>> readCSV(const std::string& fileName) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

std::vector<std::vector<double>> readAllCSVFiles(const std::string& directoryPath) {
    
    std::vector<std::vector<double>> allData(NUM_COORDS, std::vector<double>(0, 0.0));

    for (const auto& entry : fs::directory_iterator(directoryPath))  {
        if (entry.path().extension() == ".csv") {
            if (entry.path().filename().string().find("Flight") != std::string::npos) {

                // Extract the integer value from the filename
                std::string filename = entry.path().filename().string();
                size_t flightPos = filename.find("Flight-");
                size_t dotPos = filename.find(".csv");

                std::string numberStr = filename.substr(flightPos + 7, dotPos - flightPos - 7);
                int flightNumber = std::stoi(numberStr)-1;
                std::vector<std::vector<std::string>> data = readCSV(entry.path().string());
                std::vector<double> fileData;
                for (const auto& row : data) {
                    for (const auto& cell : row) {
                        double number = std::stod(cell);
                        fileData.push_back(number);
                    }
                }
                allData[flightNumber] = fileData;
            }
        }
    }
    return allData;
}

void saveToCSV(const std::vector<double>& values, const std::string& filename) {
    std::ofstream file;
    file.open(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    for (const auto& value : values) {
        file << value << "\n";
    }

    file.close();

    std::cout << "Values saved to " << filename << std::endl;
}
